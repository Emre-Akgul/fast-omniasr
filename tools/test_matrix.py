"""Render tests/matrix.json (which tests cover which configuration) as Markdown.

  python tools/test_matrix.py            # print the tables
  python tools/test_matrix.py --check    # fail if docs/test-matrix.md is out of date
  python tools/test_matrix.py --write    # regenerate docs/test-matrix.md
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "tests" / "matrix.json"
DOC = ROOT / "docs" / "test-matrix.md"
BEGIN, END = "<!-- matrix:begin -->", "<!-- matrix:end -->"


def load(path=MATRIX):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render(matrix) -> str:
    lines = ["### Tiers", "", "| tier | when | what runs | command |", "|---|---|---|---|"]
    for name, tier in matrix["tiers"].items():
        lines.append(f"| `{name}` | {tier['when']} | {tier['runs']} | `{tier['command']}` |")
    for family, info in matrix["families"].items():
        rows = [c for c in matrix["configurations"] if c["family"] == family]
        lines += ["", f"### {family.upper()} models", "",
                  f"`{info['pinned']}` is the pinned model the scheduled tiers run; the other "
                  f"{len(info['cards']) - 1} cards ("
                  + ", ".join(f"`{c}`" for c in info["cards"] if c != info["pinned"])
                  + ") share its code paths.", "",
                  f"| backend | device | precision | `{info['pinned']}` | other cards |",
                  "|---|---|---|---|---|"]
        for row in rows:
            lines.append(f"| {row['backend']} | {row['device']} | {row['precision']} | "
                         f"{', '.join(row['pinned'])} | {', '.join(row['other'])} |")
    return "\n".join(lines) + "\n"


def splice(document: str, table: str) -> str:
    start, end = document.index(BEGIN) + len(BEGIN), document.index(END)
    return document[:start] + "\n" + table + document[end:]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    table = render(load())
    if not (args.check or args.write):
        print(table, end="")
        return 0
    document = DOC.read_text(encoding="utf-8")
    updated = splice(document, table)
    if args.write:
        DOC.write_text(updated, encoding="utf-8")
        return 0
    if updated != document:
        print(f"{DOC} is out of date; run python tools/test_matrix.py --write", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
