"""Manuscript scanning: bootstrap a skeleton claims file, audit an existing one.

extract — every number on a keyword-bearing line becomes a review-flagged stub
claim. Binding stubs to ground truth is deliberately a human (or supervised)
curation step: automating it would mean guessing which artifact a number came
from, and a wrong guess that passes is worse than no binding at all.

audit — checks an existing claims file against the manuscript it was extracted
from: (a) manuscript hash drift, (b) per-claim prose drift (the claimed digits
no longer appear on the stated lines), (c) coverage (keyword-bearing numeric
lines no claim covers).
"""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

import yaml

# A number: optional minus (ascii or unicode), digits, optional decimal part,
# optional exponent. Comma accepted as group separator or decimal mark.
NUMBER_RE = re.compile(r"[−-]?\d+(?:[.,]\d+)*(?:e[−-]?\d+)?")

# Lines worth pinning in quantitative prose. Override per project with
# --keyword-regex; the default targets empirical-paper conventions.
KEYWORD_RE = re.compile(
    r"(β|beta|\bSE\b|p\s*[=<>]|p_TOST|N\s*=|N=|\br\s*=|R²|α|alpha|per cent|%|slope|"
    r"coefficient|correlat|interaction|ceiling|match rate|mean|median|std|observations)",
    re.IGNORECASE,
)

# Numbers that are almost never claims: bare 4-digit years in citation ranges.
YEARISH_RE = re.compile(r"^(19|20)\d{2}$")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def parse_number(raw: str) -> float | None:
    """Best-effort numeric parse of a regex match.

    '188,764' -> 188764 (group separator); '0,86' -> 0.86 (decimal comma);
    '−0.059' -> -0.059 (unicode minus). Returns None when ambiguous.
    """
    s = raw.replace("−", "-")
    if "," in s and "." in s:
        s = s.replace(",", "")  # 1,234.5 style
    elif "," in s:
        parts = s.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            s = s.replace(",", "")  # 188,764 style
        elif len(parts) == 2:
            s = ".".join(parts)  # 0,86 style
        else:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def scan(lines: list[str], keyword_re: re.Pattern = KEYWORD_RE) -> list[dict]:
    """Return one record per number found on a keyword-bearing line."""
    hits = []
    for lineno, line in enumerate(lines, start=1):
        if not keyword_re.search(line):
            continue
        for m in NUMBER_RE.finditer(line):
            if YEARISH_RE.match(m.group()):
                continue
            hits.append({"line": lineno, "raw": m.group(), "text": line.strip()})
    return hits


def extract(manuscript: str | Path, out_path: str | Path, keyword_regex: str | None = None,
            force: bool = False) -> dict:
    out_path = Path(out_path)
    if out_path.exists() and not force:
        # A curated claims file is the most labour-intensive artifact in the
        # workflow; a bootstrap command must never destroy it silently.
        raise FileExistsError(
            f"{out_path} already exists — refusing to overwrite curated claims. "
            f"Use --force (or force=True) if you really mean to start over."
        )
    manuscript = Path(manuscript)
    keyword_re = re.compile(keyword_regex, re.IGNORECASE) if keyword_regex else KEYWORD_RE
    lines = _read_lines(manuscript)
    hits = scan(lines, keyword_re)

    claims, counter = [], {}
    for hit in hits:
        value = parse_number(hit["raw"])
        if value is None:
            continue
        stem = f"line{hit['line']:04d}"
        counter[stem] = counter.get(stem, 0) + 1
        claims.append({
            "id": f"{stem}_{counter[stem]}",
            "text_snippet": hit["text"][:120],
            "manuscript_lines": [hit["line"]],
            "kind": "unclassified",
            "value": value,
            "comparison": "abs",
            "tolerance": 0.0005,
            "review": True,
            "notes": "extracted stub — bind to a ground-truth artifact, then set review: false",
        })

    doc = {
        "meta": {
            "generated": str(date.today()),
            "manuscript": str(manuscript),
            "manuscript_sha256": _sha256(manuscript),
            "keyword_regex": keyword_re.pattern,
            "extraction_regex": NUMBER_RE.pattern,
            "n_claims": len(claims),
            "note": "Skeleton produced by `claimpin extract`. Every claim starts review: true "
                    "(skipped loudly). Curate bindings; never let an unbound claim pass silently.",
        },
        "claims": claims,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True, width=100)
    return doc


def _digit_groups(text: str) -> list[str]:
    return [m.group() for m in NUMBER_RE.finditer(text) if not YEARISH_RE.match(m.group())]


def audit(claims_doc: dict, manuscript: str | Path, keyword_regex: str | None = None) -> dict:
    """Drift + coverage report for an existing claims file."""
    manuscript = Path(manuscript)
    keyword_re = re.compile(keyword_regex, re.IGNORECASE) if keyword_regex else KEYWORD_RE
    lines = _read_lines(manuscript)
    meta = claims_doc.get("meta", {})

    hash_drift = None
    recorded = meta.get("manuscript_sha256")
    if recorded:
        current = _sha256(manuscript)
        if current != recorded:
            hash_drift = {"recorded": recorded, "current": current}

    drifted = []
    covered_lines: set[int] = set()
    for claim in claims_doc["claims"]:
        claim_lines = claim.get("manuscript_lines") or []
        covered_lines.update(claim_lines)
        snippet = claim.get("text_snippet", "")
        groups = _digit_groups(snippet)
        if not claim_lines or not groups:
            continue
        text = " ".join(lines[ln - 1] for ln in claim_lines if 0 < ln <= len(lines))
        missing = [g for g in groups if g not in text]
        if missing:
            drifted.append({"id": claim["id"], "lines": claim_lines, "missing_digits": missing,
                            "snippet": snippet})

    unbound = [
        {"line": hit["line"], "text": hit["text"][:120]}
        for hit in scan(lines, keyword_re)
        if hit["line"] not in covered_lines and parse_number(hit["raw"]) is not None
    ]
    # one entry per line, not per number
    seen: set[int] = set()
    unbound = [u for u in unbound if not (u["line"] in seen or seen.add(u["line"]))]

    return {
        "manuscript": str(manuscript),
        "hash_drift": hash_drift,
        "drifted_claims": drifted,
        "unbound_lines": unbound,
        "ok": hash_drift is None and not drifted,
    }
