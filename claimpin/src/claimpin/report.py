"""Render a verify() result: terminal summary, JSON, and a static HTML page
that puts each claim beside the number that earns it."""
from __future__ import annotations

import html
import json
from pathlib import Path

STATUS_ORDER = ("fail", "error", "skip", "pass")

ANSI = {"pass": "\033[32m", "fail": "\033[31m", "error": "\033[35m", "skip": "\033[33m", "end": "\033[0m"}


def text_summary(result: dict, color: bool = False) -> str:
    s = result["summary"]
    lines = [
        f"claimpin verify — {result['claims_file']}",
        f"  {s['pass']} passed, {s['fail']} failed, {s['error']} errored, {s['skip']} skipped (review)",
    ]
    for status in ("fail", "error", "skip"):
        for it in result["items"]:
            if it["status"] != status:
                continue
            tag = f"{ANSI[status]}{status.upper():5}{ANSI['end']}" if color else f"{status.upper():5}"
            loc = f" (lines {', '.join(map(str, it['lines']))})" if it.get("lines") else ""
            lines.append(f"  {tag} {it['id']}{loc}: {it['message']}")
    lines.append("  OK" if result["ok"] else "  NOT OK")
    return "\n".join(lines)


def write_json(result: dict, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)


_CSS = """
body{font-family:Georgia,serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#1a1a1a}
h1{font-size:1.4rem} .meta{color:#666;font-size:.85rem;margin-bottom:1.5rem}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th{text-align:left;border-bottom:2px solid #333;padding:.4rem .6rem;font-family:Helvetica,Arial,sans-serif}
td{border-bottom:1px solid #ddd;padding:.4rem .6rem;vertical-align:top}
td.num{font-family:Consolas,monospace;white-space:nowrap}
.snippet{color:#444;font-style:italic}
.pass{background:#eaf5ea}.fail{background:#fbe3e3}.error{background:#f3e3fb}.skip{background:#fdf6df}
.badge{font-family:Helvetica,Arial,sans-serif;font-weight:bold;font-size:.75rem}
.summary{font-family:Helvetica,Arial,sans-serif;margin:.5rem 0 1rem}
"""


def write_html(result: dict, path: str | Path, title: str = "claimpin report") -> None:
    s = result["summary"]
    rows = []
    items = sorted(result["items"], key=lambda it: STATUS_ORDER.index(it["status"]))
    for it in items:
        truth = it.get("truth")
        rows.append(
            f"<tr class='{it['status']}'>"
            f"<td class='badge'>{it['status'].upper()}</td>"
            f"<td><b>{html.escape(str(it['id']))}</b>"
            f"<div class='snippet'>{html.escape(str(it.get('snippet') or ''))}</div></td>"
            f"<td class='num'>{html.escape(str(it.get('claimed', '')))}</td>"
            f"<td class='num'>{'' if truth is None else f'{truth:.6g}'}</td>"
            f"<td>{html.escape(str(it.get('message', '')))}</td></tr>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='meta'>claims file: {html.escape(result['claims_file'])}<br>"
        f"project root: {html.escape(result['project_root'])}</div>"
        f"<div class='summary'>{s['pass']} passed &middot; {s['fail']} failed &middot; "
        f"{s['error']} errored &middot; {s['skip']} skipped (review)</div>"
        "<table><tr><th>Status</th><th>Claim</th><th>Claimed</th><th>Ground truth</th><th>Detail</th></tr>"
        + "".join(rows) + "</table></body></html>"
    )
    Path(path).write_text(page, encoding="utf-8")
