"""claimpin command line: extract / verify / audit."""
from __future__ import annotations

import argparse
import json
import sys

from .extract import audit, extract
from .report import text_summary, write_html, write_json
from .runner import verify
from .schema import load_claims_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="claimpin",
        description="Pin every numeric claim in a manuscript to reproducible ground truth.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="bootstrap a skeleton claims.yaml from a manuscript")
    p_extract.add_argument("manuscript")
    p_extract.add_argument("-o", "--out", default="claims.yaml")
    p_extract.add_argument("--keyword-regex", default=None,
                           help="override the keyword filter for claim-bearing lines")
    p_extract.add_argument("--force", action="store_true",
                           help="overwrite an existing claims file (otherwise refused)")

    p_verify = sub.add_parser("verify", help="resolve every claim and run custom checks")
    p_verify.add_argument("claims", nargs="?", default="claims.yaml")
    p_verify.add_argument("--project-root", default=None,
                          help="base for relative sources (default: the claims file's directory)")
    p_verify.add_argument("--ops", default=None, help="path to a per-project ops.py plugin")
    p_verify.add_argument("--json", dest="json_out", default=None, help="write full report as JSON")
    p_verify.add_argument("--html", dest="html_out", default=None, help="write claim-beside-number HTML report")
    p_verify.add_argument("--quiet", action="store_true", help="summary line only")

    p_audit = sub.add_parser("audit", help="check claims file vs manuscript: drift + coverage")
    p_audit.add_argument("claims", nargs="?", default="claims.yaml")
    p_audit.add_argument("--manuscript", default=None,
                         help="override meta.manuscript (resolved relative to the claims file)")
    p_audit.add_argument("--keyword-regex", default=None)
    p_audit.add_argument("--json", dest="json_out", default=None)

    args = parser.parse_args(argv)

    if args.command == "extract":
        try:
            doc = extract(args.manuscript, args.out, keyword_regex=args.keyword_regex,
                          force=args.force)
        except FileExistsError as exc:
            print(f"claimpin extract: {exc}", file=sys.stderr)
            return 2
        print(f"claimpin extract — {args.manuscript}")
        print(f"  {doc['meta']['n_claims']} stub claims written to {args.out} (all review: true)")
        print("  next: bind each stub to a ground-truth artifact, then `claimpin verify`")
        return 0

    if args.command == "verify":
        result = verify(args.claims, project_root=args.project_root, ops_module=args.ops)
        if args.json_out:
            write_json(result, args.json_out)
        if args.html_out:
            write_html(result, args.html_out)
        if args.quiet:
            s = result["summary"]
            print(f"{s['pass']} passed, {s['fail']} failed, {s['error']} errored, {s['skip']} skipped")
        else:
            print(text_summary(result, color=sys.stdout.isatty()))
        return 0 if result["ok"] else 1

    if args.command == "audit":
        from pathlib import Path

        doc = load_claims_file(args.claims)
        manuscript = args.manuscript or doc["meta"].get("manuscript")
        if not manuscript:
            print("audit: no manuscript in meta and none given via --manuscript", file=sys.stderr)
            return 2
        manuscript_path = Path(manuscript)
        if not manuscript_path.is_absolute():
            candidate = Path(args.claims).resolve().parent / manuscript_path
            manuscript_path = candidate if candidate.exists() else manuscript_path
        result = audit(doc, manuscript_path, keyword_regex=args.keyword_regex)
        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
        print(f"claimpin audit — {result['manuscript']}")
        if result["hash_drift"]:
            print("  MANUSCRIPT HASH DRIFT: the manuscript changed since extraction")
        for d in result["drifted_claims"]:
            print(f"  DRIFT {d['id']} (lines {d['lines']}): digits {d['missing_digits']} no longer on stated lines")
        print(f"  {len(result['unbound_lines'])} keyword-bearing numeric lines not covered by any claim")
        for u in result["unbound_lines"][:15]:
            print(f"    line {u['line']}: {u['text']}")
        if len(result["unbound_lines"]) > 15:
            print(f"    ... and {len(result['unbound_lines']) - 15} more")
        print("  OK" if result["ok"] else "  NOT OK")
        return 0 if result["ok"] else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
