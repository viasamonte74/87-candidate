"""Truth/candidate separation (Run2 F1, W2b): the matched baseline sees only the public
contract. Proven twice: statically (import graph) and at runtime (evaluator-only modules
blocked in a fresh interpreter). Reference methods, by contrast, do reach truth code; that
is why their role is "reference" and their rows are conformance evidence only."""

import ast
import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sn87_provenonce.bundle import fixture_run
from sn87_provenonce.classes import BINDINGS
from sn87_provenonce.pilot.contracts import capsule_from_fixture
from sn87_provenonce.scoring import Integrity, score_response
from sn87_provenonce.simulation.type_c import FIXTURES

SRC = Path(__file__).parents[1] / "src"
EVALUATOR_ONLY = {
    "sn87_provenonce.institutional_v02.references",
    "sn87_provenonce.pilot.reference",
    "sn87_provenonce.simulation.type_c_reference",
}


def _source(name: str, src: Path = SRC) -> Path | None:
    path = src.joinpath(*name.split("."))
    return path / "__init__.py" if path.is_dir() else (
        path.with_suffix(".py") if path.with_suffix(".py").exists() else None)


def lazy_exports(package: str, src: Path = SRC) -> dict[str, str] | None:
    """A package's PEP 562 ``_EXPORTS`` map (name -> submodule), or None if it has none."""
    path = _source(package, src)
    for node in ast.parse(path.read_text()).body if path and path.name == "__init__.py" else []:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_EXPORTS"
                                                for t in node.targets):
            pairs = ast.literal_eval(node.value).items()
            return {f"{package}.{k}": f"{package}.{v}" for k, v in pairs}
    return None


# The only shape a lazy package ``__init__`` may take. It is the one module allowed to call
# ``import_module``, so it is checked against this template instead of being exempt: the
# docstring and the literal ``_EXPORTS`` map may vary, nothing else may. Its ``__getattr__``
# then loads exactly ``{package}.{_EXPORTS[name]}``, which the walk follows through the map.
# changes here are security-relevant; reviewer must approve explicitly
LAZY_INIT_TEMPLATE = '''
from importlib import import_module

_EXPORTS = {}
__all__ = sorted(_EXPORTS)


def __dir__() -> list[str]:
    return __all__


def __getattr__(name: str) -> object:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
'''
# Blocked by prefix: the module and every submodule (importlib.util, importlib.machinery, ...).
DYNAMIC_IMPORTS = {"importlib", "runpy", "pkgutil", "builtins", "imp", "zipimport"}
DYNAMIC_NAMES = {"exec", "eval", "compile", "__import__", "__builtins__", "__loader__",
                 "__spec__", "globals", "vars"}
DYNAMIC_ATTRS = {"import_module", "__import__", "__getattr__", "__getattribute__", "__dict__",
                 "__builtins__", "__globals__", "__subclasses__", "__code__", "__closure__",
                 "__loader__", "__spec__", "exec_module", "load_module", "module_from_spec",
                 "find_spec", "spec_from_file_location", "spec_from_loader"}
SYS_IMPORT_STATE = {"modules", "meta_path", "path_hooks", "path_importer_cache"}
# Candidate-reachable code (``miners/`` and ``baselines``) may import only these non-package
# modules; anything else, stdlib or not, fails closed (core #73). Pure typing and annotations
# are all the real tree needs. Adding an entry is security-relevant: reviewer must approve.
CANDIDATE_STDLIB_ALLOWLIST = {"__future__", "typing", "collections.abc"}
CANDIDATE_MODULES = {"baselines", "miners"}
# Helper modules a candidate module reaches (transitively) are checked too, with this wider,
# equally explicit list: exactly what the real helpers (canonical, protocol, contracts) import,
# none of it a loader. Anything else fails closed. Reviewer must approve additions.
HELPER_STDLIB_ALLOWLIST = CANDIDATE_STDLIB_ALLOWLIST | {
    "datetime", "hashlib", "json", "re", "struct", "unicodedata", "math", "enum",
    "dataclasses", "pydantic"}
# Attribute-access builtins that take an object and a string key: with a bound module or ``sys``
# as the object they reach any attribute by a computed name, which the walk cannot follow.
GETATTR_FAMILY = {"getattr", "setattr", "delattr", "hasattr"}
ATTRGETTER_NAMES = {"attrgetter", "methodcaller"}


def _lazy_init_shape(tree: ast.Module) -> str:
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    for node in body:
        if isinstance(node, ast.Assign) and [getattr(t, "id", "") for t in node.targets] == [
                "_EXPORTS"]:
            node.value = ast.Dict(keys=[], values=[])
    return ast.dump(ast.Module(body=body, type_ignores=[]))


def _check_lazy_init(name: str, tree: ast.Module) -> None:
    if _lazy_init_shape(tree) != _lazy_init_shape(ast.parse(LAZY_INIT_TEMPLATE)):
        raise AssertionError(f"lazy package {name} deviates from the PEP 562 template")


def _chain(node: ast.Attribute) -> tuple[str, list[str]] | None:
    attrs = []
    while isinstance(node, ast.Attribute):
        attrs.append(node.attr)
        node = node.value
    return (node.id, attrs[::-1]) if isinstance(node, ast.Name) else None


def _alias_leaves(value: ast.AST) -> list[ast.Name]:
    """Names an assignment value hands on unchanged (``q = p``, ``q = p if c else r``,
    ``q, r = p, s``). A call result (``f(p)``) or an attribute (``p.x``) is not an alias."""
    if isinstance(value, ast.Name):
        return [value]
    kids = (value.elts if isinstance(value, (ast.Tuple, ast.List, ast.Set)) else
            [value.body, value.orelse] if isinstance(value, ast.IfExp) else
            value.values if isinstance(value, ast.BoolOp) else
            [value.value] if isinstance(value, (ast.Starred, ast.Await)) else [])
    return [leaf for kid in kids for leaf in _alias_leaves(kid)]


def _is_module(module: str, attr: str) -> bool:
    """True if ``module.attr`` is itself a module (``typing.sys``): a re-exported module is a
    way round the allowlist. Resolved against the real interpreter, not by name."""
    try:
        return isinstance(getattr(importlib.import_module(module), attr, None), type(sys))
    except ImportError:
        return False


def closure(module: str, src: Path = SRC) -> set[str]:
    """Every sn87_provenonce module a module can load, including package __init__ files.

    Static and fail-closed. It follows import statements (absolute, relative, package
    ``__init__`` files), resolves ``from pkg import name`` on a lazy package through its
    ``_EXPORTS`` map, follows attribute chains on bound module names (``import a.b`` then
    ``a.b.f``), and raises on what it cannot prove: a chain that steps into a module this
    file did not import or through a lazy package (``import a.b`` then ``a.lazy.name``), a
    name a lazy package does not export (its own ``import_module``, ``__getattr__``,
    ``_EXPORTS``), a lazy package object escaping by name or as a chain's end, ``import *``
    from this package, the import machinery (``importlib``, ``runpy``, ``pkgutil``,
    ``builtins``, ``sys.modules`` and friends), and ``exec``/``eval``/``compile``/
    ``__import__``/``globals``/``vars``. The lazy ``__init__`` is not exempt: it must match
    ``LAZY_INIT_TEMPLATE`` exactly. The runtime tests below remain the backstop for the
    exercised baseline and candidate paths.
    """
    seen, todo, done = set(), [module], set()
    reach: set[str] = set()  # modules reachable from miners/ or baselines (candidate status)

    while todo:
        name = todo.pop()
        parts = name.split(".")
        direct = parts[:1] == ["sn87_provenonce"] and parts[1:2] and (
            parts[1] in CANDIDATE_MODULES)
        in_reach = bool(direct) or name in reach
        if (name, in_reach) in done:
            continue
        done.add((name, in_reach))
        seen.add(name)

        def push(target: str, in_reach: bool = in_reach) -> None:
            todo.append(target)
            if in_reach:
                reach.add(target)

        for i in range(1, len(parts)):
            push(".".join(parts[:i]))
        path = _source(name, src)
        tree = ast.parse(path.read_text())
        if lazy_exports(name, src) is not None:
            _check_lazy_init(name, tree)
            continue

        def fail(why: str, name: str = name) -> None:
            raise AssertionError(f"{why} in {name}: separation cannot be proven")

        bound: dict[str, str] = {}  # local name -> sn87_provenonce module it refers to
        sys_names = {"sys"} if any(isinstance(n, ast.alias) and n.name == "sys"
                                   for n in ast.walk(tree)) else set()

        is_candidate = bool(direct)
        allow = CANDIDATE_STDLIB_ALLOWLIST if is_candidate else HELPER_STDLIB_ALLOWLIST
        std_bound: dict[str, str] = {}  # local name -> allowlisted non-package module
        for node in ast.walk(tree):
            if in_reach:
                if isinstance(node, ast.Import):
                    outside = [a.name for a in node.names if not a.name.startswith(
                        "sn87_provenonce") and a.name not in allow]
                    if is_candidate and not outside:  # candidates import symbols, not modules
                        outside = [f"{a.name} (module object; import its names)"
                                   for a in node.names if not a.name.startswith(
                                       "sn87_provenonce")]
                        if outside:
                            fail(f"bare import of {outside[0]}")
                    std_bound.update({(a.asname or a.name): a.name for a in node.names
                                      if not a.name.startswith("sn87_provenonce")})
                elif isinstance(node, ast.ImportFrom) and not node.level and not (
                        node.module or "").startswith("sn87_provenonce"):
                    outside = [] if node.module in allow else [node.module]
                    for a in node.names:
                        if not outside and (a.name in DYNAMIC_ATTRS or _is_module(
                                node.module, a.name)):
                            fail(f"import of module or loader {node.module}.{a.name}")
                else:
                    outside = []
                if outside:
                    fail(f"import of {outside[0]} outside the stdlib allowlist")
            if isinstance(node, ast.alias) and node.name.split(".")[0] in DYNAMIC_IMPORTS:
                fail(f"import of {node.name}")
            if isinstance(node, ast.Name) and node.id in DYNAMIC_NAMES:
                fail(f"use of {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in DYNAMIC_ATTRS:
                fail(f"use of .{node.attr}")
            if isinstance(node, ast.ImportFrom) and (
                    node.module or "").split(".")[0] in DYNAMIC_IMPORTS:
                fail(f"import from {node.module}")
            if isinstance(node, ast.ImportFrom) and node.module == "sys" and (
                    {a.name for a in node.names} & (SYS_IMPORT_STATE | {"*"})):
                fail("import of sys import state")
            if isinstance(node, ast.Import):
                sys_names |= {a.asname for a in node.names if a.name == "sys" and a.asname}
            if (isinstance(node, ast.Attribute) and node.attr in SYS_IMPORT_STATE
                    and isinstance(node.value, ast.Name) and node.value.id in sys_names):
                fail(f"use of sys.{node.attr}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("sn87_provenonce"):
                        continue
                    if lazy_exports(alias.name, src) is not None:
                        fail(f"binding lazy package {alias.name}")
                    push(alias.name)
                    root = alias.name.split(".")[0]
                    bound[alias.asname or root] = alias.name if alias.asname else root
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                base = ".".join(parts[: len(parts) - node.level + (path.name == "__init__.py")])
                module_name = f"{base}.{node.module}" if node.module else base
            else:
                module_name = node.module or ""
            if not module_name.startswith("sn87_provenonce"):
                continue
            push(module_name)
            lazy = lazy_exports(module_name, src)
            for alias in node.names:
                if alias.name == "*":
                    fail(f"from {module_name} import *")
                target = f"{module_name}.{alias.name}"
                if lazy is not None and target in lazy:
                    push(lazy[target])
                elif _source(target, src):
                    if lazy_exports(target, src) is not None:
                        fail(f"binding lazy package {target}")
                    push(target)
                    bound[alias.asname or alias.name] = target
                elif lazy is not None:
                    fail(f"{alias.name} is not a lazy export of {module_name}")

        # Scope-blind bindings fail closed: a name bound to an sn87_provenonce module may not be
        # bound to anything else anywhere in the file (any scope, import, assignment, def, arg).
        import_targets: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    local = a.asname or a.name.split(".")[0]
                    import_targets.setdefault(local, set()).add(
                        a.name if a.asname else a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    import_targets.setdefault(a.asname or a.name, set()).add(
                        f"{node.level}:{node.module}.{a.name}")
        for local in bound:
            if len(import_targets.get(local, ())) > 1:
                fail(f"name {local} is bound to more than one module")
        for node in ast.walk(tree):
            rebound = (
                node.id if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
                else node.arg if isinstance(node, ast.arg)
                else node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                    ast.ClassDef, ast.ExceptHandler))
                else node.name if isinstance(node, (ast.MatchAs, ast.MatchStar))
                else None)
            names = {rebound} | set(getattr(node, "names", []) if isinstance(
                node, (ast.Global, ast.Nonlocal)) else ())
            if names & set(bound):
                fail(f"name {sorted(names & set(bound))[0]} bound to a module is rebound")

        # Getattr family and aliasing (core #73). A bound module or ``sys`` may only appear as the
        # base of an attribute chain: handing it to getattr/setattr/delattr/hasattr (any key,
        # including a concatenated string), to attrgetter, or to any other value position (an
        # alias such as ``q = p``) hides what it reaches. ``sys`` is tracked like a bound module.
        tracked = set(bound) | sys_names | set(std_bound)
        for node in ast.walk(tree):  # an allowlisted module's attributes may not be modules
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id in std_bound and _is_module(
                        std_bound[node.value.id], node.attr)):
                fail(f"{node.value.id}.{node.attr} is a module reached through an allowlisted "
                     "module")
        for node in ast.walk(tree):  # loader names imported from any sn87_provenonce module
            if isinstance(node, ast.ImportFrom) and any(
                    a.name in DYNAMIC_ATTRS for a in node.names):
                fail("import of a loader or computed-access name")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else (
                    fn.attr if isinstance(fn, ast.Attribute) else "")
                if fname in ATTRGETTER_NAMES and tracked:
                    fail(f"use of {fname} next to bound module or sys")
                first = node.args[0] if node.args else None
                while isinstance(first, ast.Attribute):
                    first = first.value
                if fname in GETATTR_FAMILY and isinstance(first, ast.Name) and (
                        first.id in tracked):
                    fail(f"{fname} on bound module {first.id}")
        based = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                    and node.id in tracked and id(node) not in based):
                continue
            if is_candidate:  # candidate code has no reason to hand a module around at all
                fail(f"bound module {node.id} used as a value")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr, ast.AugAssign)) and (
                    node.value is not None):
                for leaf in _alias_leaves(node.value):
                    if leaf.id in tracked:
                        fail(f"bound module {leaf.id} aliased by assignment")

        imported = set(bound.values()) | {
            ".".join(a.name.split(".")[:i]) for n in ast.walk(tree) if isinstance(n, ast.Import)
            for a in n.names for i in range(1, len(a.name.split(".")) + 1)}
        inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in bound and lazy_exports(
                    bound[node.id], src) is not None:
                fail(f"lazy package {bound[node.id]} used by name")
            if not isinstance(node, ast.Attribute) or id(node) in inner:
                continue
            chain = _chain(node)
            if chain is None or chain[0] not in bound:
                continue
            current = bound[chain[0]]
            for attr in chain[1]:
                if lazy_exports(current, src) is not None:
                    fail(f"attribute chain through lazy package {current}")
                if not _source(f"{current}.{attr}", src):
                    break
                current = f"{current}.{attr}"
                if current not in imported:
                    fail(f"attribute chain reaches unimported module {current}")
                push(current)
            if lazy_exports(current, src) is not None:
                fail(f"lazy package {current} escapes an attribute chain")
    return seen


def test_baseline_import_closure_excludes_evaluator_only():
    reachable = closure("sn87_provenonce.baselines")
    assert not reachable & EVALUATOR_ONLY
    if all(_source(name) for name in EVALUATOR_ONLY):  # the check can see them (full tree)
        assert closure("sn87_provenonce.classes") >= EVALUATOR_ONLY


@pytest.mark.parametrize("evasion", [
    "from sn87_provenonce.simulation import execute_state_machine",
    "import importlib\nimportlib.import_module('sn87_provenonce.pilot.reference')",
    "from importlib import import_module",
    "x = __import__('sn87_provenonce.pilot.reference')",
    "import sn87_provenonce.simulation as s",
    "from sn87_provenonce import simulation",
])
@pytest.mark.requires_private_executors
def test_closure_catches_lazy_and_dynamic_evasions(evasion, tmp_path):
    """Each way of reaching truth that the static walk could miss is caught or flagged.

    Probes are written into a private copy of the source tree, so parallel or interrupted
    runs never leave a stray module in the repository.
    """
    src = tmp_path / "src"
    shutil.copytree(SRC, src, ignore=shutil.ignore_patterns("__pycache__"))
    probe = src / "sn87_provenonce" / "_separation_probe.py"
    probe.write_text("from sn87_provenonce import baselines\n" + evasion + "\n")
    try:
        try:
            leaked = closure("sn87_provenonce._separation_probe", src) & EVALUATOR_ONLY
        except AssertionError:
            return
        assert leaked, evasion
    finally:
        assert not (SRC / "sn87_provenonce" / "_separation_probe.py").exists()


def _tmp_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    shutil.copytree(SRC, src, ignore=shutil.ignore_patterns("__pycache__"))
    return src


GAP_EVASIONS = {
    # attribute chains after a dotted import
    "chain_through_lazy": "import sn87_provenonce.pilot.contracts\n"
                          "sn87_provenonce.simulation.execute_state_machine",
    "chain_lazy_escapes": "import sn87_provenonce.simulation.type_c\n"
                          "f = getattr(sn87_provenonce.simulation, 'execute_state_machine')",
    "chain_unimported_submodule": "import sn87_provenonce.pilot.contracts\n"
                                  "sn87_provenonce.pilot.reference",
    "chain_from_bound_package": "from sn87_provenonce import pilot\npilot.reference",
    "chain_alias": "import sn87_provenonce.pilot as p\np.reference",
    # import *
    "star_lazy": "from sn87_provenonce.simulation import *",
    "star_plain": "from sn87_provenonce.pilot import *",
    # the lazy package's own loader used by name
    "lazy_import_module": "from sn87_provenonce.simulation import import_module\n"
                          "import_module('sn87_provenonce.pilot.reference')",
    "lazy_getattr": "from sn87_provenonce.simulation import __getattr__\n"
                    "__getattr__('execute_state_machine')",
    "lazy_exports_map": "from sn87_provenonce.simulation import _EXPORTS",
    "lazy_getattr_chain": "import sn87_provenonce.simulation.type_c\n"
                          "sn87_provenonce.simulation.__getattr__('execute_state_machine')",
    # sys.modules, runpy, pkgutil
    "sys_modules": "import sys\nsys.modules['sn87_provenonce.simulation']",
    "sys_alias_modules": "import sys as s\ns.modules",
    "from_sys_modules": "from sys import modules",
    "sys_meta_path": "import sys\nsys.meta_path",
    "runpy": "import runpy\nrunpy.run_module('sn87_provenonce.pilot.reference')",
    "pkgutil": "import pkgutil",
    "pkgutil_from": "from pkgutil import resolve_name",
    "builtins": "import builtins",
    # exec / eval / __import__ and friends
    "exec": "exec('from sn87_provenonce.pilot import reference')",
    "eval": "eval('1')",
    "exec_alias": "run = exec",
    "compile": "compile('x', 'y', 'exec')",
    "dunder_import": "f = __import__",
    "globals": "g = globals()",
    "module_dict": "import sn87_provenonce.pilot.contracts\n"
                   "sn87_provenonce.pilot.contracts.__dict__",
    # importlib submodules, in any form (#72 review)
    "importlib_util_from": "from importlib.util import find_spec, module_from_spec\n"
                           "spec = find_spec('sn87_provenonce.pilot.reference')\n"
                           "module = module_from_spec(spec)\n"
                           "spec.loader.exec_module(module)",
    "importlib_machinery": "import importlib.machinery",
    "importlib_metadata_from": "from importlib.metadata import entry_points",
    "runpy_submodule_from": "from runpy import run_path",
    # scope-blind bindings (#72 review): a nested rebinding must not shadow the module one
    "nested_rebinding": "import sn87_provenonce.pilot.contracts\n"
                        "def f():\n"
                        "    import sn87_provenonce.pilot.reference as sn87_provenonce\n"
                        "sn87_provenonce.pilot.contracts",
    "nested_rebinding_reviewer": "import sn87_provenonce.pilot.contracts\n"
                                 "def f():\n"
                                 "    import os as sn87_provenonce\n"
                                 "    return sn87_provenonce.path\n"
                                 "sn87_provenonce.pilot.contracts",
    "assigned_rebinding": "from sn87_provenonce import pilot\n"
                          "def f(pilot):\n"
                          "    return pilot.reference",
    "two_modules_one_name": "from sn87_provenonce.pilot import contracts as c\n"
                            "from sn87_provenonce.pilot import reference as c",
}
# Core #73 closes the module-level cases; aliasing of non-module objects in general stays out of
# scope, and any legitimate future ``globals()``/``compile()`` needs a reviewed exception.


@pytest.mark.parametrize("evasion", GAP_EVASIONS.values(), ids=GAP_EVASIONS)
@pytest.mark.requires_private_executors
def test_closure_raises_on_import_graph_gaps_in_miners_path(evasion, tmp_path):
    """#41: each gap the old walk could not see now fails closed, planted in the miners path
    (a tmp copy of ``miners/witness_ic.py``) and in a standalone probe module."""
    src = _tmp_src(tmp_path)
    witness = src / "sn87_provenonce" / "miners" / "witness_ic.py"
    witness.write_text(witness.read_text() + "\n" + evasion + "\n")
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce.miners", src)
    probe = src / "sn87_provenonce" / "_gap_probe.py"
    probe.write_text(evasion + "\n")
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce._gap_probe", src)
    assert not (SRC / "sn87_provenonce" / "_gap_probe.py").exists()


# Core #73: (probe source, message the guard must raise). The message is asserted so that each
# probe is pinned to its own guard: reverting one guard cannot be masked by another one.
GETATTR_PROBES = {
    "getattr_string_key": ("import sn87_provenonce.pilot.contracts as p\n"
                           "getattr(p, 'reference')", "getattr on bound module p"),
    "getattr_concatenated_key": ("import sn87_provenonce.pilot.contracts as p\n"
                                 "getattr(p, 'ref' + 'erence')", "getattr on bound module p"),
    "getattr_sys_modules": ("import sys\ngetattr(sys, 'mod' + 'ules')",
                            "getattr on bound module sys"),
    "getattr_package_submodule": ("import sn87_provenonce.pilot.contracts\n"
                                  "getattr(sn87_provenonce.pilot, 'reference')",
                                  "getattr on bound module sn87_provenonce"),
    "hasattr_bound": ("from sn87_provenonce.pilot import contracts\n"
                      "hasattr(contracts, 'x')", "hasattr on bound module contracts"),
    "setattr_bound": ("from sn87_provenonce.pilot import contracts\n"
                      "setattr(contracts, 'x', 1)", "setattr on bound module contracts"),
    "delattr_bound": ("from sn87_provenonce.pilot import contracts\n"
                      "delattr(contracts, 'x')", "delattr on bound module contracts"),
    "getattr_sys_alias": ("import sys as s\ngetattr(s, 'modules')",
                          "getattr on bound module s"),
    "attrgetter_module": ("import operator\nfrom sn87_provenonce.pilot import contracts\n"
                          "operator.attrgetter('reference')(contracts)",
                          "use of attrgetter next to bound module"),
    "attrgetter_from": ("from operator import attrgetter\nimport sys\n"
                        "attrgetter('mod' + 'ules')(sys)",
                        "use of attrgetter next to bound module"),
}
ALIAS_PROBES = {
    "alias_hop": ("import sn87_provenonce.pilot as p\nq = p\nq.reference",
                  "bound module p aliased by assignment"),
    "alias_hop_from_import": ("from sn87_provenonce import pilot\nq = pilot\nq.reference",
                              "bound module pilot aliased by assignment"),
    "alias_tuple": ("import sn87_provenonce.pilot as p\nq, r = p, 1\nq.reference",
                    "bound module p aliased by assignment"),
    "alias_conditional": ("import sn87_provenonce.pilot as p\nq = p if 1 else None\n"
                          "q.reference", "bound module p aliased by assignment"),
    "alias_walrus": ("import sn87_provenonce.pilot as p\n(q := p).reference",
                     "bound module p aliased by assignment"),
    "alias_annotated": ("import sn87_provenonce.pilot as p\nq: object = p\nq.reference",
                        "bound module p aliased by assignment"),
    "alias_sys": ("import sys\nq = sys\nq.modules", "bound module sys aliased by assignment"),
}
# Candidate-only: modules outside the allowlist, planted in miners/witness_ic.py and baselines.py.
ALLOWLIST_PROBES = {
    "pydoc_locate": "import pydoc\npydoc.locate('sn87_provenonce.pilot.reference')",
    "pydoc_from": "from pydoc import locate",
    "pickle_loads": "import pickle\npickle.loads(b'')",
    "pickle_from": "from pickle import loads",
    "os_import": "import os",
    "json_import": "import json",
    "sys_import": "import sys",
    "typing_submodule_style": "import collections",
    "operator_import": "import operator",
    "functools_from": "from functools import partial",
    "unlisted_third_party": "import numpy",
}
VALUE_PROBES = {
    "module_passed": ("from sn87_provenonce.pilot import contracts\nlen(contracts)",
                      "bound module contracts used as a value"),
}


def _plant(tmp_path: Path, relative: str, probe: str) -> Path:
    src = _tmp_src(tmp_path)
    target = src / "sn87_provenonce" / relative
    target.write_text(target.read_text() + "\n" + probe + "\n")
    return src


@pytest.mark.parametrize("probe,message", GETATTR_PROBES.values(), ids=GETATTR_PROBES)
def test_closure_fails_closed_on_getattr_family(probe, message, tmp_path):
    src = _plant(tmp_path, "miners/witness_ic.py", probe)
    # In the candidate path several guards overlap (allowlist, module-as-value), so only
    # fail-closed is asserted there; the standalone module isolates this probe's own guard.
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce.miners", src)
    (src / "sn87_provenonce" / "_getattr_probe.py").write_text(probe + "\n")
    with pytest.raises(AssertionError, match=message):
        closure("sn87_provenonce._getattr_probe", src)


@pytest.mark.parametrize("probe,message", ALIAS_PROBES.values(), ids=ALIAS_PROBES)
def test_closure_fails_closed_on_name_aliasing(probe, message, tmp_path):
    src = _plant(tmp_path, "miners/witness_ic.py", probe)
    # In the candidate path several guards overlap (allowlist, module-as-value), so only
    # fail-closed is asserted there; the standalone module isolates this probe's own guard.
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce.miners", src)
    (src / "sn87_provenonce" / "_alias_probe.py").write_text(probe + "\n")
    with pytest.raises(AssertionError, match=message):
        closure("sn87_provenonce._alias_probe", src)


@pytest.mark.parametrize("probe,message", VALUE_PROBES.values(), ids=VALUE_PROBES)
@pytest.mark.parametrize("target,root", [("miners/witness_ic.py", "sn87_provenonce.miners"),
                                         ("baselines.py", "sn87_provenonce.baselines")])
def test_candidate_code_may_not_pass_a_module_as_a_value(probe, message, target, root, tmp_path):
    src = _plant(tmp_path, target, probe)
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure(root, src)
    (src / "sn87_provenonce" / "miners" / "_probe_mod.py").write_text(probe + "\n")
    with pytest.raises(AssertionError, match=message):
        closure("sn87_provenonce.miners._probe_mod", src)


@pytest.mark.parametrize("probe", ALLOWLIST_PROBES.values(), ids=ALLOWLIST_PROBES)
@pytest.mark.parametrize("target,root", [("miners/witness_ic.py", "sn87_provenonce.miners"),
                                         ("miners/__init__.py", "sn87_provenonce.miners"),
                                         ("baselines.py", "sn87_provenonce.baselines")])
def test_candidate_stdlib_import_outside_allowlist_fails_closed(probe, target, root, tmp_path):
    src = _plant(tmp_path, target, probe)
    with pytest.raises(AssertionError, match="outside the stdlib allowlist"):
        closure(root, src)


# Review follow-ups. (1) An allowlisted module may not hand back another module (``typing.sys``)
# or be used as a bare object; (2) computed-access dunders beyond getattr; (3) candidate status
# propagates to the helpers a candidate reaches. Each probe is pinned to its own guard.
REEXPORT_PROBES = {
    "typing_sys_from": ("from typing import sys\ngetattr(sys, 'modules')",
                        "import of module or loader typing.sys"),
    "typing_sys_attr_helper": ("import json\njson.decoder", "json.decoder is a module"),
    "bare_allowlisted_alias": ("import json\nq = json\nq.decoder",
                               "bound module json aliased by assignment"),
    "bare_allowlisted_getattr": ("import hashlib\ngetattr(hashlib, 'sys')",
                                 "getattr on bound module hashlib"),
}
DUNDER_PROBES = {
    "getattribute_builtins": (
        "import sn87_provenonce.pilot.contracts as p\n"
        "p.__getattribute__('__builtins__')['__import__']('sn87_provenonce.pilot.reference')",
        "use of .__getattribute__"),
    "function_globals": ("from sn87_provenonce.pilot.contracts import capsule_from_fixture\n"
                         "capsule_from_fixture.__globals__",
                         "use of .__globals__"),
    "import_getattribute_name": ("from sn87_provenonce.pilot.contracts import __getattribute__",
                                 "loader or computed-access name"),
    "subclasses": ("object.__subclasses__()", "use of .__subclasses__"),
}
HELPER_PROBES = {
    "helper_pydoc": "import pydoc\npydoc.locate('sn87_provenonce.pilot.reference')",
    "helper_pickle": "from pickle import loads",
    "helper_os": "import os",
    "helper_sys": "import sys",
}


@pytest.mark.parametrize("probe,message", REEXPORT_PROBES.values(), ids=REEXPORT_PROBES)
def test_allowlisted_modules_may_not_reexport_modules(probe, message, tmp_path):
    src = _plant(tmp_path, "miners/witness_ic.py", probe)
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce.miners", src)
    # A helper reached from a candidate: isolates the helper-tier guard (bare imports allowed).
    (src / "sn87_provenonce" / "_helper.py").write_text(probe + "\n")
    (src / "sn87_provenonce" / "baselines.py").write_text(
        "from sn87_provenonce import _helper\n")
    with pytest.raises(AssertionError, match=message):
        closure("sn87_provenonce.baselines", src)


@pytest.mark.parametrize("probe,message", DUNDER_PROBES.values(), ids=DUNDER_PROBES)
def test_closure_fails_closed_on_computed_access_dunders(probe, message, tmp_path):
    src = _plant(tmp_path, "miners/witness_ic.py", probe)
    with pytest.raises(AssertionError, match="separation cannot be proven"):
        closure("sn87_provenonce.miners", src)
    (src / "sn87_provenonce" / "_dunder_probe.py").write_text(probe + "\n")
    with pytest.raises(AssertionError, match=message):
        closure("sn87_provenonce._dunder_probe", src)


@pytest.mark.parametrize("probe", HELPER_PROBES.values(), ids=HELPER_PROBES)
@pytest.mark.parametrize("entry", ["miners/witness_ic.py", "baselines.py"])
def test_candidate_restrictions_follow_into_imported_helpers(probe, entry, tmp_path):
    src = _plant(tmp_path, entry, "from sn87_provenonce import _helper")
    (src / "sn87_provenonce" / "_helper.py").write_text(probe + "\n")
    root = "sn87_provenonce.miners" if entry.startswith("miners") else "sn87_provenonce.baselines"
    with pytest.raises(AssertionError, match="outside the stdlib allowlist"):
        closure(root, src)
    # Not reached from a candidate: the same helper alone is unrestricted.
    assert "sn87_provenonce._helper" in closure("sn87_provenonce._helper", src)


def test_candidate_status_reaches_a_helper_already_visited_without_it(tmp_path):
    """Order must not matter: a helper first walked as non-candidate is re-checked once a
    candidate is found to reach it."""
    src = _plant(tmp_path, "miners/witness_ic.py", "from sn87_provenonce import _helper")
    (src / "sn87_provenonce" / "_helper.py").write_text("import pydoc\n")
    (src / "sn87_provenonce" / "_both.py").write_text(
        "from sn87_provenonce import _helper\nfrom sn87_provenonce import miners\n")
    with pytest.raises(AssertionError, match="outside the stdlib allowlist"):
        closure("sn87_provenonce._both", src)


def test_real_helpers_reached_from_candidates_need_no_extra_allowlist():
    """Zero false positives: the real helper closure passes under the helper allowlist."""
    for root in ("sn87_provenonce.baselines", "sn87_provenonce.miners"):
        assert "sn87_provenonce.protocol.v0alpha1.canonical" in closure(root)


def test_candidate_allowlisted_imports_pass(tmp_path):
    src = _plant(tmp_path, "baselines.py",
                 "from typing import Any\nfrom collections.abc import Mapping\n"
                 "from __future__ import annotations")
    # ``from __future__`` mid-file is not valid Python, but the walk only parses; keep it valid:
    (src / "sn87_provenonce" / "baselines.py").write_text(
        "from __future__ import annotations\nfrom typing import Any\n"
        "from collections.abc import Mapping\n")
    assert "sn87_provenonce.baselines" in closure("sn87_provenonce.baselines", src)


def test_non_candidate_modules_keep_ordinary_stdlib_imports(tmp_path):
    """The allowlist is scoped to miners/ and baselines: reference code may use the stdlib."""
    src = _tmp_src(tmp_path)
    (src / "sn87_provenonce" / "_plain_probe.py").write_text("import json, pickle\n")
    assert "sn87_provenonce._plain_probe" in closure("sn87_provenonce._plain_probe", src)


def test_real_tree_candidate_imports_are_inside_the_allowlist():
    """Zero false positives: every candidate-path import in the real tree is allowlisted or a
    package import, so the allowlist needed no entry beyond what the tree already uses."""
    used = set()
    for path in [SRC / "sn87_provenonce" / "baselines.py",
                 *(SRC / "sn87_provenonce" / "miners").glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and not node.level:
                used.add(node.module)
            elif isinstance(node, ast.Import):
                used |= {a.name for a in node.names}
    assert {m for m in used if not m.startswith("sn87_provenonce")} <= CANDIDATE_STDLIB_ALLOWLIST


def test_real_lazy_init_matches_template():
    """The real ``simulation/__init__`` is checked, not exempt: it matches the template."""
    init = SRC / "sn87_provenonce" / "simulation" / "__init__.py"
    _check_lazy_init("sn87_provenonce.simulation", ast.parse(init.read_text()))


LAZY_INIT_TAMPERING = {
    "module_level_load": ("\n", "\nimport_module('sn87_provenonce.simulation.type_c_reference')\n"),
    "getattr_target": ('import_module(f"{__name__}.{_EXPORTS[name]}")',
                       'import_module("sn87_provenonce.pilot.reference")'),
    "static_import": ("from importlib import import_module",
                      "from importlib import import_module\nfrom . import type_c_reference"),
    "extra_function": ("def __dir__", "def load(n):\n    return import_module(n)\n\n\n"
                                      "def __dir__"),
}


@pytest.mark.parametrize("old,new", LAZY_INIT_TAMPERING.values(), ids=LAZY_INIT_TAMPERING)
def test_closure_raises_on_tampered_lazy_init(old, new, tmp_path):
    src = _tmp_src(tmp_path)
    init = src / "sn87_provenonce" / "simulation" / "__init__.py"
    text = init.read_text()
    tampered = text + new if old == "\n" else text.replace(old, new, 1)
    assert tampered != text
    init.write_text(tampered)
    (src / "sn87_provenonce" / "_lazy_probe.py").write_text(
        "from sn87_provenonce.simulation import FIXTURES\n")
    with pytest.raises(AssertionError, match="deviates from the PEP 562 template"):
        closure("sn87_provenonce._lazy_probe", src)


@pytest.mark.requires_private_executors
def test_dotted_import_chains_that_are_imported_still_resolve(tmp_path):
    """Fail-closed does not mean blind: an explicitly imported chain is followed."""
    src = _tmp_src(tmp_path)
    (src / "sn87_provenonce" / "_chain_probe.py").write_text(
        "import sn87_provenonce.pilot.reference\nsn87_provenonce.pilot.reference\n")
    assert "sn87_provenonce.pilot.reference" in closure("sn87_provenonce._chain_probe", src)


@pytest.mark.requires_private_executors
def test_closure_follows_the_given_source_root(tmp_path):
    """Modules that exist only in the tmp copy, reached through a lazy export, a relative import
    and a ``from package import submodule`` edge, must be resolved there: the root is threaded
    through every lookup."""
    src = tmp_path / "src"
    shutil.copytree(SRC, src, ignore=shutil.ignore_patterns("__pycache__"))
    sim = src / "sn87_provenonce" / "simulation"
    (sim / "_only_in_copy.py").write_text("from . import type_c_reference\n")
    init = sim / "__init__.py"
    export = '_EXPORTS = {\n    "only": "_only_in_copy",'
    init.write_text(init.read_text().replace("_EXPORTS = {", export, 1))
    probe = src / "sn87_provenonce" / "_root_probe.py"
    (src / "sn87_provenonce" / "pilot" / "_copy_only.py").write_text(
        "from sn87_provenonce.pilot import reference\n")
    probe.write_text("from sn87_provenonce.simulation import only\n"
                     "from sn87_provenonce.pilot import _copy_only\n")
    reached = closure("sn87_provenonce._root_probe", src)
    assert "sn87_provenonce.pilot._copy_only" in reached
    assert "sn87_provenonce.simulation._only_in_copy" in reached
    assert "sn87_provenonce.simulation.type_c_reference" in reached


BLOCKED_RUN = """
import sys
BLOCKED = %r
class Block:
    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED:
            raise ImportError("evaluator-only module blocked: " + name)
sys.meta_path.insert(0, Block())
from sn87_provenonce import baselines
from sn87_provenonce.institutional_v02.fixtures import build_case
from sn87_provenonce.pilot.contracts import capsule_from_fixture
from sn87_provenonce.simulation.type_c import FIXTURES
states = [baselines.ic_approval_applicability(build_case(f))["state"]
          for f in ("stale_authority", "fresh_review", "incomplete")]
states += [baselines.type_c_release(capsule_from_fixture(f))["state"] for f in FIXTURES]
assert not BLOCKED & set(sys.modules), BLOCKED & set(sys.modules)
print(",".join(states))
"""


def test_baseline_runs_with_evaluator_only_modules_blocked():
    run = subprocess.run([sys.executable, "-c", BLOCKED_RUN % EVALUATOR_ONLY],
                         capture_output=True, text=True, check=True)
    assert run.stdout.strip().startswith("FINDINGS,NO_MATERIAL_DEVIATION,")
    # classes imports without the executors; only a call that needs truth fails, loudly.
    reference_run = BLOCKED_RUN.replace(
        "from sn87_provenonce import baselines",
        "from sn87_provenonce.classes import IC_APPROVAL_APPLICABILITY as IC\n"
        "from sn87_provenonce import baselines\n"
        "try:\n    IC.reference({})\nexcept RuntimeError as e:\n    print(e)\n")
    blocked = subprocess.run([sys.executable, "-c", reference_run % EVALUATOR_ONLY],
                             capture_output=True, text=True)
    assert blocked.returncode == 0, blocked.stderr
    assert "private reference executor not available" in blocked.stdout


@pytest.mark.parametrize("class_id", sorted(BINDINGS))
def test_baseline_matches_truth_and_null_result_is_kept(class_id):
    """First Light and the pilot are structurally easy: the public-contract baseline scores
    like the references. The bundle keeps that null result instead of inventing headroom."""
    bundle = fixture_run(BINDINGS[class_id])
    roles = {m["method_id"]: m["role"] for m in bundle["methods"]}
    assert roles == {"state_machine": "reference", "relational": "reference",
                     "public_contract_baseline": "baseline"}
    assert bundle["baseline"]["state"] == "CONFIGURED"
    assert bundle["baseline"]["public_contract_only"] is True
    assert bundle["scores"]["public_contract_baseline"]["estimate"] == "1"
    assert set(bundle["row"]["weights"]) == {"state_machine", "relational"}
    assert bundle["comparison"] == {m: {"estimate_delta": "0", "result": "NULL_NO_HEADROOM"}
                                    for m in ("state_machine", "relational")}
    assert bundle["cost"]["public_contract_baseline"]["oracle_time"]["value"] == "NA"


@pytest.mark.parametrize("fixture", FIXTURES, ids=lambda f: f.fixture_id)
def test_type_c_baseline_on_every_public_fixture(fixture):
    binding = BINDINGS["TYPE-C-RELEASE"]
    capsule = capsule_from_fixture(fixture)
    truth = binding.reference(capsule)
    result = score_response(binding, capsule, truth, binding.baseline(capsule),
                            Integrity(*[True] * 6))
    assert result["valid"] and binding.baseline(capsule)["state"] == truth["state"]


def test_ic_baseline_abstains_when_governing_versions_tie():
    from sn87_provenonce.baselines import ic_approval_applicability
    from sn87_provenonce.canonical import evidence_commitment
    from sn87_provenonce.institutional_v02.fixtures import build_case

    capsule = build_case("stale_authority")
    versions = [e for e in capsule["events"] if e["kind"] == "POLICY_VERSION"]
    # Both versions govern every action, with the same effective time: an ambiguous tie.
    versions[1]["at"] = versions[0]["at"]
    versions[1]["effective_at"] = versions[0]["effective_at"]
    capsule["evidence_commitment"] = evidence_commitment(capsule)
    assert ic_approval_applicability(capsule)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"


def test_miners_import_closure_excludes_evaluator_only():
    reachable = closure("sn87_provenonce.miners")
    assert "sn87_provenonce.miners.witness_ic" in reachable
    assert "sn87_provenonce.miners.witness_type_c" in reachable
    assert not reachable & EVALUATOR_ONLY


CANDIDATE_RUN = BLOCKED_RUN.replace(
    "from sn87_provenonce import baselines",
    "from sn87_provenonce import miners\n"
    "candidate = miners.for_class('IC-APPROVAL-APPLICABILITY')['approval_witness']",
).split("states = [")[0] + """states = [candidate(build_case(f))["state"]
          for f in ("stale_authority", "fresh_review", "incomplete")]
assert not BLOCKED & set(sys.modules), BLOCKED & set(sys.modules)
print(",".join(states))
"""


def test_candidate_runs_with_evaluator_only_modules_blocked():
    run = subprocess.run([sys.executable, "-c", CANDIDATE_RUN % EVALUATOR_ONLY],
                         capture_output=True, text=True, check=True)
    assert run.stdout.strip() == "FINDINGS,NO_MATERIAL_DEVIATION,INSUFFICIENT_EVIDENCE_ABSTAIN"
