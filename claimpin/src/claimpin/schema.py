"""Load and validate a claims.yaml file.

The file has two top-level keys:

  meta:    manuscript path, manuscript_sha256, ops_module, extraction regexes —
           everything about WHERE the claims came from.
  claims:  the list of claims themselves.

Each claim:

  id                unique slug (test id, report key)
  text_snippet      the manuscript prose being pinned (verbatim)
  manuscript_lines  line number(s) where the snippet lives
  kind              free-form tag (n, coefficient, percent, p_value, ...)
  value             the number the manuscript asserts
  comparison        abs (default) | lt | gt
  tolerance         absolute tolerance for comparison: abs
  binds_to          op + params resolving to ground truth (see resolver.py)
  review            true = no committed ground truth; SKIPPED loudly, never
                    silently passed
  notes             free-form provenance
"""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED = ("id", "value")
VALID_COMPARISONS = ("abs", "lt", "gt")


class ClaimsError(ValueError):
    pass


def load_claims_file(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise ClaimsError(f"claims file not found: {path}")
    with open(path, encoding="utf-8-sig") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict) or "claims" not in doc:
        raise ClaimsError(f"{path}: expected a mapping with a top-level `claims` list")
    doc.setdefault("meta", {})
    validate(doc["claims"], str(path))
    return doc


def validate(claims: list, origin: str = "claims.yaml") -> None:
    if not isinstance(claims, list):
        raise ClaimsError(f"{origin}: `claims` must be a list")
    seen: set[str] = set()
    for i, claim in enumerate(claims):
        where = f"{origin}: claims[{i}]"
        if not isinstance(claim, dict):
            raise ClaimsError(f"{where} is not a mapping")
        for field in REQUIRED:
            if field not in claim:
                raise ClaimsError(f"{where} ({claim.get('id', '?')}): missing `{field}`")
        cid = claim["id"]
        if cid in seen:
            raise ClaimsError(f"{where}: duplicate id {cid!r}")
        seen.add(cid)
        comparison = claim.get("comparison", "abs")
        if comparison not in VALID_COMPARISONS:
            raise ClaimsError(f"{where} ({cid}): comparison {comparison!r} not in {VALID_COMPARISONS}")
        if not claim.get("review", False) and "binds_to" not in claim:
            raise ClaimsError(
                f"{where} ({cid}): no `binds_to` and not flagged `review: true` — "
                f"a claim must be bound to ground truth or explicitly awaiting it"
            )
        binds = claim.get("binds_to")
        if binds is not None and not isinstance(binds, dict):
            raise ClaimsError(f"{where} ({cid}): `binds_to` must be a mapping")
