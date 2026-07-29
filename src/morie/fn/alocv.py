# morie.fn -- function file (rootcoder007/morie)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Output verification gate (Alammar Ch 7)."""

from ._richresult import RichResult

__all__ = ["alammar_output_verification"]


def alammar_output_verification(response, criteria, verifier_model):
    """passed = every criterion's verdict is PASS.

    The verifier is a callable (response, criterion) -> verdict; any
    verdict other than the literal strings "PASS"/"FAIL" is refused,
    because a verifier that answers in prose has not answered.

    References: Alammar and Grootendorst, Ch 7.
    """
    if not callable(verifier_model):
        raise ValueError("verifier_model must be callable "
                         "(response, criterion) -> 'PASS' | 'FAIL'.")
    crits = [str(c) for c in criteria]
    if not crits:
        raise ValueError("no criteria supplied; an empty gate passes "
                         "everything and verifies nothing.")
    verdicts = {}
    for c in crits:
        v = verifier_model(str(response), c)
        if v not in ("PASS", "FAIL"):
            raise ValueError(
                f"verifier returned {v!r} for {c!r}; only 'PASS' or "
                "'FAIL' count as answers.")
        verdicts[c] = v
    passed = all(v == "PASS" for v in verdicts.values())
    return RichResult(payload={
        "passed": passed, "verdicts": verdicts,
        "failed_criteria": [c for c, v in verdicts.items() if v == "FAIL"],
        "estimate": float(passed), "n": len(crits),
        "method": "Criterion-gated output verification (Alammar Ch 7)"})


def cheatsheet():
    return "alocv: all-criteria PASS gate, prose verdicts refused"
