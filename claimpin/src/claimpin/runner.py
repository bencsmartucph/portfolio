"""Verify orchestration: resolve every claim, run every custom check.

Statuses:
  pass   bound claim whose ground truth matches within tolerance
  fail   bound claim whose ground truth does NOT match
  skip   review-flagged claim (no committed ground truth) — loud, never silent
  error  binding raised (missing file, bad path, unresolvable op)

`fail` and `error` both make the run unsuccessful; they are reported
separately because they mean different things (drifted number vs broken
binding).
"""
from __future__ import annotations

import traceback
from pathlib import Path

import claimpin

from .context import Context, load_plugin
from .resolver import check, resolve
from .schema import load_claims_file


def verify(
    claims_path: str | Path,
    project_root: str | Path | None = None,
    ops_module: str | Path | None = None,
) -> dict:
    claims_path = Path(claims_path).resolve()
    doc = load_claims_file(claims_path)
    meta = doc["meta"]
    claims = doc["claims"]

    # Precedence: explicit argument > meta key (relative to the claims file's
    # directory) > the claims file's directory itself.
    base = claims_path.parent
    if project_root is None:
        project_root = base / meta["project_root"] if "project_root" in meta else base
    ctx = Context(Path(project_root))

    if ops_module is None and "ops_module" in meta:
        ops_module = base / meta["ops_module"]
    if ops_module is not None:
        load_plugin(Path(ops_module))

    claims_by_id = {c["id"]: c for c in claims}
    items: list[dict] = []

    for claim in claims:
        item = {
            "id": claim["id"],
            "item_type": "claim",
            "snippet": claim.get("text_snippet", ""),
            "lines": claim.get("manuscript_lines", []),
            "claimed": claim.get("value"),
            "truth": None,
        }
        if claim.get("review", False):
            item["status"] = "skip"
            item["message"] = f"REVIEW: no committed ground truth — {claim.get('notes', 'see claims file')}"
        else:
            try:
                truth = resolve(claim["binds_to"], ctx, claims_by_id=claims_by_id)
                ok, msg = check(claim, truth)
                item["truth"] = truth
                item["status"] = "pass" if ok else "fail"
                item["message"] = msg
            except Exception as exc:  # noqa: BLE001 — every claim must report, not abort the run
                item["status"] = "error"
                item["message"] = f"{type(exc).__name__}: {exc}"
                item["trace"] = traceback.format_exc(limit=3)
        items.append(item)

    for name, fn in claimpin.CHECKS.items():
        item = {"id": name, "item_type": "check", "snippet": (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""}
        try:
            fn(ctx)
            item["status"] = "pass"
            item["message"] = "ok"
        except AssertionError as exc:
            item["status"] = "fail"
            item["message"] = str(exc) or "assertion failed"
        except Exception as exc:  # noqa: BLE001
            item["status"] = "error"
            item["message"] = f"{type(exc).__name__}: {exc}"
            item["trace"] = traceback.format_exc(limit=3)
        items.append(item)

    summary = {s: sum(1 for it in items if it["status"] == s) for s in ("pass", "fail", "skip", "error")}
    return {
        "claims_file": str(claims_path),
        "project_root": str(ctx.project_root),
        "meta": meta,
        "items": items,
        "summary": summary,
        "ok": summary["fail"] == 0 and summary["error"] == 0,
    }
