#!/usr/bin/env python3
"""Compare playbook options vs right-only options (default bag)."""
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location(
    "vco", r"d:\code\qqq-dip-playbook\vboost_compare_options.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules["vco"] = m
spec.loader.exec_module(m)

rows = m.align(m.fetch_yahoo("QQQ"), m.fetch_yahoo("^VXN"))
episodes = [
    ("2020-02-01", "2020-12-31", "2020", ["2020-03-23", "2020-06-08", "2020-12-31"], "rolling60"),
    ("2022-01-03", "2023-12-29", "2022-roll", ["2022-06-16", "2022-10-14", "2023-12-29"], "rolling60"),
    ("2022-01-03", "2023-12-29", "2022-peak", ["2022-06-16", "2022-10-14", "2023-12-29"], "cycle_peak"),
    ("2025-02-01", "2025-12-31", "2025", ["2025-04-08", "2025-06-10", "2025-12-31"], "rolling60"),
]

out = {"episodes": []}
print(f"{'sample':10} {'policy':12} {'end':>8} {'dd':>8} {'calls':>5}  notes")
for start, end, label, marks, hm in episodes:
    row = {"label": label, "h_mode": hm, "policies": {}}
    for pol in ("playbook", "right_only"):
        r = m.run_full(rows, start, end, "default", marks, hm, option_policy=pol)
        calls = [b for b in r["buys"] if str(b["kind"]).startswith("call")]
        left = [b["tier"] for b in calls if b["tier"] != "R1"]
        r1 = [b for b in calls if b["tier"] == "R1"]
        note = f"left={','.join(left) or '-'} R1={'yes' if r1 else 'no'}"
        if r1:
            note += f"@{r1[0]['date']} ${r1[0]['usd']:.0f}"
        print(
            f"{label:10} {pol:12} {r['end']['ret_pct']:+7.1f}% {r['max_dd_pct']:7.1f}% "
            f"{len(calls):5d}  {note}"
        )
        row["policies"][pol] = {
            "end_ret_pct": r["end"]["ret_pct"],
            "max_dd_pct": r["max_dd_pct"],
            "call_count": len(calls),
            "marks": {k: v["ret_pct"] for k, v in r["marks"].items()},
            "buys_calls": calls,
        }
    pb = row["policies"]["playbook"]
    ro = row["policies"]["right_only"]
    row["delta_right_minus_playbook"] = {
        "end_ret_pp": round(ro["end_ret_pct"] - pb["end_ret_pct"], 2),
        "max_dd_pp": round(ro["max_dd_pct"] - pb["max_dd_pct"], 2),
    }
    print(
        f"{'':10} {'Δ right-pb':12} {row['delta_right_minus_playbook']['end_ret_pp']:+7.1f}pp "
        f"{row['delta_right_minus_playbook']['max_dd_pp']:+7.1f}pp"
    )
    out["episodes"].append(row)

path = r"d:\code\qqq-dip-playbook\right-only-options-compare.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote", path)
