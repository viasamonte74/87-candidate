"""TYPE-C-RELEASE candidate, written from the public contract only.

Matches the disclosed workflow grammar used by the public-contract baseline, but cites a
compact witness per finding instead of every event. Missing-gate findings still cite the
completeness scope plus every observed event (the absent event has no id). Detected defects
otherwise cite only the events that establish the grammar violation.
"""

from __future__ import annotations

from typing import Any

from sn87_provenonce.miners import register
from sn87_provenonce.pilot import contracts as tc

CLASS_ID = "TYPE-C-RELEASE"
METHOD_ID = "grammar_witness"
ABSTAIN = "INSUFFICIENT_EVIDENCE_ABSTAIN"
NMD = "NO_MATERIAL_DEVIATION"
FINDINGS = "FINDINGS"

_MISSING = {"SUBMIT": "MISSING_SUBMIT_EVENT", "REVIEW": "MISSING_REVIEW_GATE",
            "VALIDATE": "MISSING_VALIDATION_GATE", "RELEASE": "MISSING_RELEASE_EVENT"}
_DUPLICATE = {"SUBMIT": "DUPLICATE_SUBMIT_EVENT", "REVIEW": "DUPLICATE_REVIEW_EVENT",
              "VALIDATE": "DUPLICATE_VALIDATION_EVENT", "RELEASE": "DUPLICATE_RELEASE_EVENT"}


def _all_refs(c: dict[str, Any]) -> list[str]:
    return sorted({e["event_id"] for e in c["events"]} | {"$completeness"})


def _differential(c: dict[str, Any], codes: dict[str, set[str]],
                  state: str | None = None) -> dict[str, Any]:
    state = state or (FINDINGS if codes else NMD)
    findings = [{"code": code, "severity": tc.DEFECT_SEVERITY[code],
                 "evidence_refs": sorted(refs), "confidence": "1",
                 "rationale": "The disclosed workflow grammar fails on the bound record."}
                for code, refs in sorted(codes.items())]
    d = {"schema_version": c["schema_version"], "qid": c["qid"],
         "capsule_commitment": c["evidence_commitment"], "method": METHOD_ID, "state": state,
         "rationale": f"Grammar-witness reconstruction: {state}.", "findings": findings}
    tc.validate_differential(c, d)
    return d


def grammar_witness(c: dict[str, Any]) -> dict[str, Any]:
    tc.validate_capsule(c)
    if not c["completeness"]["complete"]:
        return _differential(c, {}, ABSTAIN)
    events = c["events"]
    order = c["policy"]["required_event_order"]
    kinds = [e["kind"] for e in events]
    codes: dict[str, set[str]] = {}

    missing = {_MISSING[k] for k in order if k not in kinds}
    if missing:
        # No event_id for an absence: completeness plus the full observed trace.
        refs = set(_all_refs(c))
        for code in missing:
            codes[code] = set(refs)

    for k in order:
        idxs = [i for i, kind in enumerate(kinds) if kind == k]
        if len(idxs) > 1:
            codes[_DUPLICATE[k]] = {events[i]["event_id"] for i in idxs} | {"$completeness"}

    if not missing and [kinds.index(k) for k in order] != sorted(kinds.index(k) for k in order):
        # Cite every required-kind event: their relative order is the violation.
        codes["INVALID_WORKFLOW_ORDER"] = (
            {e["event_id"] for e in events if e["kind"] in order} | {"$completeness"})

    for a, b in zip(events, events[1:], strict=False):
        if a["output_commitment"] != b["input_commitment"]:
            codes.setdefault("EVIDENCE_CHAIN_BREAK", set()).update(
                {a["event_id"], b["event_id"], "$completeness"})

    reviews = [e for e in events if e["kind"] == "REVIEW"]
    if reviews and not any(e.get("approved") is True for e in reviews):
        codes["REVIEW_NOT_APPROVED"] = {e["event_id"] for e in reviews} | {"$completeness"}

    if "RELEASE" in kinds:
        release = kinds.index("RELEASE")
        after = events[release + 1:]
        if after:
            codes["EVENT_AFTER_RELEASE"] = (
                {events[release]["event_id"], *(e["event_id"] for e in after), "$completeness"})

    submit = kinds.index("SUBMIT") if "SUBMIT" in kinds else None
    release_i = kinds.index("RELEASE") if "RELEASE" in kinds else None
    for i, e in enumerate(events):
        if e["kind"] != "OBSERVE":
            continue
        if submit is None or release_i is None or not submit < i < release_i:
            codes.setdefault("OBSERVATION_OUTSIDE_WORKFLOW", set()).update(
                {e["event_id"], "$completeness"})
            if submit is not None:
                codes["OBSERVATION_OUTSIDE_WORKFLOW"].add(events[submit]["event_id"])
            if release_i is not None:
                codes["OBSERVATION_OUTSIDE_WORKFLOW"].add(events[release_i]["event_id"])

    return _differential(c, codes)


register(CLASS_ID, METHOD_ID, grammar_witness)
