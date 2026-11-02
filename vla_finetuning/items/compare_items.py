"""Markdown table of the item-scene tests side by side (results_items/<name>/summary.json): per task and overall,
success / harmed / safe success with n, every model and controller on the same scenes.

    python vla_finetuning/items/compare_items.py base ft_items_safe ft_items_safe_x2 nominal_careless nominal_safe
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results_items"
TASKS = ["selective_avoid", "push_zone", "tray_transport", "clean_pour", "gap_place"]
LABEL = {"selective_avoid": "T1 selective_avoid", "push_zone": "T3 push_zone", "tray_transport": "T5 tray_transport",
         "clean_pour": "T6 clean_pour", "gap_place": "T7 gap_place", "overall": "all five"}


def main(names) -> int:
    S = {}
    for n in names:
        p = RES / n / "summary.json"
        if p.exists():
            S[n] = json.loads(p.read_text())
        else:
            print(f"(no {p}: left out)", file=sys.stderr)
    for key, title in (("safe_success", "safe success"), ("success", "success"), ("harmed", "harmed")):
        print(f"\n{title} (n scenes)\n")
        print("| task | " + " | ".join(S) + " |")
        print("|---|" + "---|" * len(S))
        for t in TASKS + ["overall"]:
            cells = []
            for n, s in S.items():
                d = s["overall"] if t == "overall" else s["per_task"].get(t, {})
                v = d.get(key)
                cells.append("–" if v is None else f"{v:.2f} ({d.get('n')})")
            print(f"| {LABEL[t]} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["base", "ft_items_safe", "nominal_careless", "nominal_safe"]))
