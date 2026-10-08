#!/usr/bin/env python3
"""Localhost miner server one miner behind a stdlib HTTP byte boundary.

  uv run python scripts/miner_serve.py --role candidate|baseline --port N [--host 127.0.0.1]

LOCAL ONLY. NOT A BITTENSOR AXON. NO CHAIN, NO WALLET, NO KEYS, NO SERVE REGISTRATION.

The server binds loopback only (a non-loopback ``--host`` is refused at startup) and wraps the
same miner callables the Provenonce staging harness uses: ``candidate`` is
``miners.witness_ic.approval_witness``, ``baseline`` is the public-contract baseline algorithm
run as a miner.

Wire format (identical bytes to the in-process path, so scoring is identical):

  GET  /health  -> 200 JSON {"status":"ok","role","method_id","class_id","cmt_commitment",
                   "cmt_verified":true,"label"}
  POST /cmt     request body  = canonical capsule bytes, exactly ``canonical_bytes(capsule)``
                                (the public challenge instance the validator derives from the
                                CMT-named generator, sent as canonical bytes);
                                Content-Type is ignored; body limit 1 MiB.
                response body = ``canonical_bytes(miner(parse_canonical(request)))`` with
                                Content-Type application/octet-stream, status 200. The response
                                is the miner's answer unchanged (finding / no material
                                difference / abstain, per its ``state`` field).
                optional request header ``X-CMT-Commitment: sha256:...``: if present it must
                equal this server's verified CMT commitment, else 409.
  errors        JSON body {"error": CODE, "detail": str}; 400 MALFORMED_REQUEST (not canonical
                JSON / not a valid capsule / broken evidence commitment), 400
                CMT_GRAMMAR_MISMATCH, 409 CMT_COMMITMENT_MISMATCH, 411 LENGTH_REQUIRED, 413
                BODY_TOO_LARGE, 404, 405, 500 MINER_FAILED (the miner raised; the server keeps
                serving). The server never crashes on a bad request.

What a miner can verify (and what it cannot): at startup it compiles the CMT and verifies its
commitment, profile commitment and class exactly as the validator side does
(fail closed). Per request it verifies the capsule against the ``institution/0.2`` contract
(``validate_capsule``, which recomputes ``evidence_commitment``) and that the capsule grammar
equals the CMT's ``generator_version``. A capsule carries no CMT commitment itself, so binding
a request to a specific CMT is by the optional header above. A miner cannot verify that the
validator derived the instance from the committed seed or generator, nor the truth.

SIGTERM and SIGINT shut the server down cleanly (exit 0). Calls are serialized by a lock.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import signal
import socket
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sn87_provenonce import miners  # noqa: E402
from sn87_provenonce.baselines import BASELINES  # noqa: E402
from sn87_provenonce.canonical import canonical_bytes, parse_canonical  # noqa: E402
from sn87_provenonce.classes import IC_APPROVAL_APPLICABILITY  # noqa: E402
from sn87_provenonce.cmt import (  # noqa: E402
    CompiledCMT,
    cmt_canonical_bytes,
    cmt_commitment,
    compile_cmt,
    verify_cmt,
)
from sn87_provenonce.institutional_v02.contracts import (  # noqa: E402
    validate_capsule,
    validate_differential,
)

LABEL = "LOCAL ONLY, FICTIONAL FIXTURES, NO CHAIN, NOT AN AXON"
MAX_BODY = 1 << 20
CANDIDATE = "approval_witness"  # the same ids the staging harness uses
BASELINE_MINER = "baseline_miner"
ROLES = {"candidate": CANDIDATE, "baseline": BASELINE_MINER}


class StagingError(RuntimeError):
    """CMT verification failed closed."""


def verify_compiled(compiled: CompiledCMT, binding=IC_APPROVAL_APPLICABILITY) -> str:
    """Same checks as the validator side's CMT verification (kept local so this script imports
    nothing from bittensor): commitment reproduces, profile and class match the binding."""
    manifest = compiled.manifest
    commitment = compiled.compatibility.manifest_commitment
    try:
        verify_cmt(cmt_canonical_bytes(manifest), commitment)
    except ValueError as error:
        raise StagingError(f"CMT_HASH_VERIFICATION_FAILED: {error}") from error
    if cmt_commitment(manifest) != commitment:
        raise StagingError("CMT_HASH_VERIFICATION_FAILED: commitment mismatch")
    if manifest.evaluation.profile_commitment != binding.profile.commitment:
        raise StagingError("CMT_PROFILE_COMMITMENT_MISMATCH")
    if manifest.identity.class_id != binding.class_id:
        raise StagingError("CMT_CLASS_MISMATCH")
    return commitment


class RefusedHost(ValueError):
    """The requested bind host is not loopback."""


def _resolve_loopback(host: str) -> None:
    """Refuse ``localhost`` unless every address it resolves to is loopback."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as error:
        raise RefusedHost(f"NON_LOOPBACK_HOST_REFUSED: {host} does not resolve: {error}") from error
    addrs = {info[4][0].split("%")[0] for info in infos}
    if not addrs or not all(ipaddress.ip_address(a).is_loopback for a in addrs):
        raise RefusedHost(f"NON_LOOPBACK_HOST_REFUSED: {host} resolves to {sorted(addrs)}")


def require_loopback(host: str) -> str:
    if host == "localhost":
        _resolve_loopback(host)
    else:
        try:
            ok = ipaddress.ip_address(host).is_loopback
        except ValueError:
            ok = False
        if not ok:
            raise RefusedHost(f"NON_LOOPBACK_HOST_REFUSED: {host}")
    return host


def role_method(role: str):
    if role == "candidate":
        return miners.for_class(IC_APPROVAL_APPLICABILITY.class_id)[CANDIDATE]
    if role == "baseline":
        return BASELINES[IC_APPROVAL_APPLICABILITY.class_id]
    raise ValueError(f"unknown role: {role}")


class MinerServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, TimeoutError):  # includes socket.timeout: a stalled client
            sys.stderr.write(f"[miner_serve {self.role}] client {client_address[0]} timed out "
                             "reading the request; dropped\n")
            return
        super().handle_error(request, client_address)

    def __init__(self, host: str, port: int, role: str, compiled: CompiledCMT | None = None):
        require_loopback(host)
        self.role, self.method_id, self.method = role, ROLES[role], role_method(role)
        self.compiled = compiled or compile_cmt()
        self.cmt_commitment = verify_compiled(self.compiled)  # fail closed, as the validator does
        self.grammar = self.compiled.manifest.renewal.generator_version
        self.lock = threading.Lock()
        if host == "::1":
            self.address_family = socket.AF_INET6
        elif ":" in host:  # e.g. ::ffff:127.0.0.1 is not bindable as AF_INET; use 127.0.0.1
            raise RefusedHost(f"UNSUPPORTED_HOST_FORM_USE_127.0.0.1_OR_::1: {host}")
        super().__init__((host, port), Handler)

    def answer(self, body: bytes) -> bytes:
        """The byte boundary: canonical capsule bytes in, canonical response bytes out."""
        capsule = parse_canonical(body)
        differential = self.method(capsule)
        validate_differential(capsule, differential)  # refuse malformed miner output
        return canonical_bytes(differential)


class Handler(BaseHTTPRequestHandler):
    server: MinerServer
    timeout = 10  # seconds; a stalled client cannot hold a thread forever
    server_version = "sn87-miner-serve/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet; no request content logged
        sys.stderr.write(f"[miner_serve {self.server.role}] {fmt % args}\n")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, detail: str = "") -> None:
        self.close_connection = True
        self._send(status, json.dumps({"error": code, "detail": detail}, sort_keys=True)
                   .encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            return self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND", self.path)
        s = self.server
        self._send(200, json.dumps({
            "status": "ok", "role": s.role, "method_id": s.method_id,
            "class_id": IC_APPROVAL_APPLICABILITY.class_id,
            "cmt_commitment": s.cmt_commitment, "cmt_verified": True, "label": LABEL},
            sort_keys=True).encode(), "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/cmt":
            return self._error(HTTPStatus.NOT_FOUND, "NOT_FOUND", self.path)
        s = self.server
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0:
                raise ValueError
        except ValueError:
            return self._error(HTTPStatus.LENGTH_REQUIRED, "LENGTH_REQUIRED")
        if length > MAX_BODY:
            if length <= 4 * MAX_BODY:  # drain so a well-behaved client sees the 413
                self.rfile.read(length)
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "BODY_TOO_LARGE",
                               f"limit {MAX_BODY}")
        body = self.rfile.read(length)
        claimed = self.headers.get("X-CMT-Commitment")
        if claimed is not None and claimed != s.cmt_commitment:
            return self._error(HTTPStatus.CONFLICT, "CMT_COMMITMENT_MISMATCH",
                               f"server commitment {s.cmt_commitment}")
        try:
            capsule = parse_canonical(body)
            validate_capsule(capsule)  # recomputes evidence_commitment
        except Exception as error:  # noqa: BLE001 - any parse/validation failure is a 400
            return self._error(HTTPStatus.BAD_REQUEST, "MALFORMED_REQUEST",
                               f"{type(error).__name__}: {error}"[:300])
        if capsule["policy"].get("grammar") != s.grammar:
            return self._error(HTTPStatus.BAD_REQUEST, "CMT_GRAMMAR_MISMATCH",
                               f"expected {s.grammar}")
        try:
            with s.lock:
                out = s.answer(body)
        except Exception as error:  # noqa: BLE001 - miner-owned failure; keep serving
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "MINER_FAILED",
                               type(error).__name__)
        self._send(200, out, "application/octet-stream")

    def do_PUT(self) -> None:  # noqa: N802
        self._error(HTTPStatus.METHOD_NOT_ALLOWED, "METHOD_NOT_ALLOWED")

    do_DELETE = do_PATCH = do_PUT  # noqa: N815


def serve(server: MinerServer) -> None:
    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--role", choices=sorted(ROLES), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)
    try:
        server = MinerServer(args.host, args.port, args.role)
    except (RefusedHost, StagingError, OSError) as error:
        print(f"MINER SERVE REFUSED: {error}", file=sys.stderr)
        return 2
    host, port = server.server_address[:2]
    print(f"{LABEL}\nrole={args.role} listening on http://{host}:{port} "
          f"cmt={server.cmt_commitment}", flush=True)
    serve(server)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
