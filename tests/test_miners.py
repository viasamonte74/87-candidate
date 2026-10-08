"""The replaceable candidates: bound as role "candidate", measured against truth and baseline."""

import pytest

from sn87_provenonce import miners
from sn87_provenonce.baselines import ic_approval_applicability as ic_baseline
from sn87_provenonce.baselines import type_c_release as tc_baseline
from sn87_provenonce.bundle import fixture_run
from sn87_provenonce.canonical import evidence_commitment
from sn87_provenonce.classes import BINDINGS, with_candidates
from sn87_provenonce.classes import IC_APPROVAL_APPLICABILITY as IC
from sn87_provenonce.institutional_v02.fixtures import build_case
from sn87_provenonce.pilot.contracts import capsule_from_fixture
from sn87_provenonce.simulation.type_c import FIXTURES

IC_CLASS = "IC-APPROVAL-APPLICABILITY"
IC_METHOD = "approval_witness"
TC_CLASS = "TYPE-C-RELEASE"
TC_METHOD = "grammar_witness"


def test_registry_and_binding():
    assert set(miners.for_class(IC_CLASS)) == {IC_METHOD}
    assert set(miners.for_class(TC_CLASS)) == {TC_METHOD}
    with pytest.raises(ValueError):
        miners.register(IC_CLASS, IC_METHOD, lambda c: c)
    bound = with_candidates(IC, miners.for_class(IC_CLASS))
    assert [bound.role_of(m) for m in bound.candidates] == ["reference", "reference", "candidate"]
    assert set(IC.candidates) == {"state_machine", "relational"} and not IC.roles  # unchanged
    with pytest.raises(ValueError):
        with_candidates(IC, {"relational": lambda c: c})


def test_measured_candidate_equals_truth_and_baseline_on_first_light_fixtures():
    """Measured, not designed: on the public First Light generator the candidate matches truth
    on every case, so against the matched baseline the result is a null (no headroom). The
    witness set is smaller than the baseline's, and still earns full evidence credit."""
    bundle = fixture_run(with_candidates(IC, miners.for_class(IC_CLASS)))
    roles = {m["method_id"]: m["role"] for m in bundle["methods"]}
    assert roles[IC_METHOD] == "candidate" and roles["public_contract_baseline"] == "baseline"
    assert bundle["methods"][2]["uid"] is None
    score = bundle["scores"][IC_METHOD]
    assert (score["admitted"], score["evaluable"], score["estimate"], score["eligible"]) == (
        12, 12, "1", True)
    assert bundle["comparison"][IC_METHOD] == {"estimate_delta": "0", "result": "NULL_NO_HEADROOM"}
    assert set(bundle["row"]["weights"]) == {"state_machine", "relational", IC_METHOD}
    assert bundle["cost"][IC_METHOD]["oracle_time"]["missing"].startswith("NOT_APPLICABLE")
    assert bundle["diagnostics"][IC_METHOD]["correct_abstentions"] == 4


def test_witness_is_smaller_than_the_baselines_and_ambiguity_is_handled_its_own_way():
    candidate = miners.for_class(IC_CLASS)[IC_METHOD]

    capsule = build_case("stale_authority")
    mine = candidate(capsule)["findings"][0]["evidence_refs"]
    theirs = ic_baseline(capsule)["findings"][0]["evidence_refs"]
    assert set(mine) < set(theirs) and len(mine) < len(capsule["events"])
    # A same-time tie of governing versions with DIFFERENT conditions is ambiguous: abstain.
    # With the SAME condition the tie is moot: the candidate decides, the baseline abstains.
    tied = build_case("stale_authority")
    versions = [e for e in tied["events"] if e["kind"] == "POLICY_VERSION"]
    versions[1]["at"], versions[1]["effective_at"] = versions[0]["at"], versions[0]["effective_at"]
    tied["evidence_commitment"] = evidence_commitment(tied)
    assert candidate(tied)["state"] == ic_baseline(tied)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"
    versions[1]["condition_commitment"] = versions[0]["condition_commitment"]
    tied["evidence_commitment"] = evidence_commitment(tied)
    assert candidate(tied)["state"] == "NO_MATERIAL_DEVIATION"
    assert ic_baseline(tied)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"
    # Measured divergence: the reference truth abstains on any tie, so this decisiveness is
    # not rewarded (detection is 0 whenever truth abstains). Kept, not tuned away.
    assert IC.reference(tied)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"


def test_ic_abstains_when_actor_bindings_break():
    candidate = miners.for_class(IC_CLASS)[IC_METHOD]
    capsule = build_case("stale_authority")
    review = next(e for e in capsule["events"] if e["kind"] == "REVIEW")
    review["actor_id"] = "not-the-policy-reviewer"
    capsule["evidence_commitment"] = evidence_commitment(capsule)
    # Baseline may still FINDINGS; hardened candidate abstains (admitted LIMITATIONS §13 gap).
    assert candidate(capsule)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"
    assert ic_baseline(capsule)["state"] == "FINDINGS"


def test_ic_abstains_when_reviewed_condition_was_never_governed():
    candidate = miners.for_class(IC_CLASS)[IC_METHOD]
    capsule = build_case("stale_authority")
    # Keep the old policy so actions still have a governing version, but publish it only
    # after the review that cites it: the condition was never governed before the review.
    review = next(e for e in capsule["events"] if e["kind"] == "REVIEW")
    old = next(e for e in capsule["events"]
               if e["kind"] == "POLICY_VERSION"
               and e["condition_commitment"] == review["condition_commitment"])
    old["at"] = review["at"]  # not strictly before the review
    capsule["evidence_commitment"] = evidence_commitment(capsule)
    assert candidate(capsule)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"
    # Baseline ignores "was ever governed" and still reports STALE_APPROVAL.
    assert ic_baseline(capsule)["state"] == "FINDINGS"


def test_ic_abstains_when_handoff_topology_breaks():
    candidate = miners.for_class(IC_CLASS)[IC_METHOD]
    capsule = build_case("fresh_review")
    handoff = next(e for e in capsule["events"] if e["kind"] == "HANDOFF")
    handoff["to_actor"] = capsule["policy"]["agents"]["origin"]  # successor never receives
    capsule["evidence_commitment"] = evidence_commitment(capsule)
    assert candidate(capsule)["state"] == "INSUFFICIENT_EVIDENCE_ABSTAIN"
    assert ic_baseline(capsule)["state"] == "NO_MATERIAL_DEVIATION"


def test_type_c_candidate_matches_baseline_states_with_tighter_witness():
    candidate = miners.for_class(TC_CLASS)[TC_METHOD]
    for fixture in FIXTURES:
        capsule = capsule_from_fixture(fixture)
        mine = candidate(capsule)
        theirs = tc_baseline(capsule)
        assert mine["state"] == theirs["state"]
        assert {f["code"] for f in mine["findings"]} == {f["code"] for f in theirs["findings"]}
        for finding in mine["findings"]:
            assert "$completeness" in finding["evidence_refs"]
            assert set(finding["evidence_refs"]) <= (
                {e["event_id"] for e in capsule["events"]} | {"$completeness"})
        if mine["state"] == "FINDINGS" and "MISSING_REVIEW_GATE" not in {
                f["code"] for f in mine["findings"]}:
            # Non-missing defects should not need to cite every event.
            for finding in mine["findings"]:
                if finding["code"].startswith("MISSING_"):
                    continue
                assert len(finding["evidence_refs"]) <= len(capsule["events"]) + 1


def test_type_c_chain_break_cites_only_the_break():
    candidate = miners.for_class(TC_CLASS)[TC_METHOD]
    fixture = next(f for f in FIXTURES if f.fixture_id == "type-c-chain-break")
    capsule = capsule_from_fixture(fixture)
    finding = next(f for f in candidate(capsule)["findings"] if f["code"] == "EVIDENCE_CHAIN_BREAK")
    baseline_refs = tc_baseline(capsule)["findings"][0]["evidence_refs"]
    assert set(finding["evidence_refs"]) < set(baseline_refs)
    assert len(finding["evidence_refs"]) == 3  # two events + $completeness


def test_existing_fixture_bundles_do_not_gain_the_candidate():
    for binding in BINDINGS.values():
        assert set(fixture_run(binding)["scores"]) == {
            "state_machine", "relational", "public_contract_baseline"}
