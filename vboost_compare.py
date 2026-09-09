#!/usr/bin/env python3
"""Compare default 40/30/30 vs V-boost 55/20/25 across QQQ dip episodes.

Episodes: 2020 COVID, 2022 bear, 2025 spring, 2026 shallow spring.

Equity-proxy model (fair for bag-split comparison):
- Deploy QQQ shares at each triggered tier using that day's close.
- VXN gates ignored for sizing (same tiers fire); options omitted so the delta is pure bag sizing.
- After -22%, also deploy R1/R2 on simple signals: R1 = first close above 20DMA after T4 with 5d high; R2 = 15 trading days later if low holds.
- Cash earns 0. Unused cash stays at face value.
- Track NAV = cash + shares * price; peak-to-trough drawdown after first buy.
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}


def fetch_qqq(start="2019-12-01", end="2026-09-10"):
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/QQQ"
        f"?period1={p1}&period2={p2}&interval=1d&events=history"
    )
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.load(resp)
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        c = q["close"][i]
        h = q["high"][i]
        lo = q["low"][i]
        if c is None or h is None or lo is None:
            continue
        rows.append(
            {
                "date": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"),
                "close": float(c),
                "high": float(h),
                "low": float(lo),
            }
        )
    return rows


DEFAULT_PCT = {
    "T1": 0.06,
    "T2": 0.08,
    "T3": 0.14,
    "T4": 0.12,
    "T5": 0.10,
    "T6": 0.10,
    "T7": 0.10,
    "R1": 0.10,
    "R2": 0.20,
}
BAG = {
    "default": {"left": 0.40, "crisis": 0.30, "right": 0.30},
    "vBoost": {"left": 0.55, "crisis": 0.20, "right": 0.25},
}
TIER_BAG = {
    "T1": "left",
    "T2": "left",
    "T3": "left",
    "T4": "left",
    "T5": "crisis",
    "T6": "crisis",
    "T7": "crisis",
    "R1": "right",
    "R2": "right",
}
MULT = {
    "T1": 0.92,
    "T2": 0.88,
    "T3": 0.82,
    "T4": 0.78,
    "T5": 0.70,
    "T6": 0.60,
    "T7": 0.50,
}


def tier_pct(tier: str, variant: str) -> float:
    base = BAG["default"][TIER_BAG[tier]]
    target = BAG[variant][TIER_BAG[tier]]
    return DEFAULT_PCT[tier] * (target / base)


def rolling_high(rows, i, lookback=60):
    window = rows[max(0, i - lookback + 1) : i + 1]
    return max(r["high"] for r in window)


def sma(rows, i, n=20):
    if i + 1 < n:
        return None
    window = rows[i - n + 1 : i + 1]
    return sum(r["close"] for r in window) / n


@dataclass
class Sim:
    variant: str
    B: float = 100_000.0
    cash: float = 100_000.0
    shares: float = 0.0
    fired: dict = field(default_factory=dict)
    buys: list = field(default_factory=list)
    nav_series: list = field(default_factory=list)
    H: float | None = None
    has_minus22: bool = False
    r1_at: int | None = None
    r1_swing_low: float | None = None
    r2_frozen: bool = False
    peak_nav: float = 100_000.0
    max_dd: float = 0.0
    first_buy_i: int | None = None


def nav(sim: Sim, px: float) -> float:
    return sim.cash + sim.shares * px


def buy_tier(sim: Sim, tier: str, i: int, row: dict):
    if tier in sim.fired:
        return
    pct = tier_pct(tier, sim.variant)
    usd = round(sim.B * pct, 2)
    if usd <= 0 or sim.cash < usd * 0.01:
        return
    usd = min(usd, sim.cash)
    px = row["close"]
    sh = usd / px
    sim.cash -= usd
    sim.shares += sh
    sim.fired[tier] = {"date": row["date"], "px": px, "usd": usd, "shares": sh}
    sim.buys.append({"tier": tier, "date": row["date"], "px": px, "usd": usd})
    if sim.first_buy_i is None:
        sim.first_buy_i = i


def run_episode(rows, start_date: str, end_date: str, variant: str, label: str):
    sim = Sim(variant=variant)
    idxs = [i for i, r in enumerate(rows) if start_date <= r["date"] <= end_date]
    if not idxs:
        raise SystemExit(f"no rows for {label}")

    for i in idxs:
        row = rows[i]
        # refresh H each day from 60d high as of today
        sim.H = rolling_high(rows, i, 60)
        H = sim.H
        c = row["close"]
        lo = row["low"]

        # reset if back to 0.98H after some firing (stop new buys but keep holdings)
        if sim.fired and c >= 0.98 * H:
            # do not void already fired; just skip new left/crisis until next episode
            pass

        # left / crisis triggers on close
        for tier, m in MULT.items():
            if tier.startswith("R"):
                continue
            if c <= m * H:
                if tier in ("T5", "T6", "T7") and not sim.has_minus22:
                    continue
                buy_tier(sim, tier, i, row)
                if tier == "T4" or c <= 0.78 * H or lo <= 0.75 * H:
                    sim.has_minus22 = True

        # intraday T4 if low hits 0.75H
        if lo <= 0.75 * H:
            sim.has_minus22 = True
            buy_tier(sim, "T4", i, row)

        # R1: after -22%, close > 20DMA and 5-day high
        if sim.has_minus22 and "R1" not in sim.fired and not sim.r2_frozen:
            ma = sma(rows, i, 20)
            if ma is not None and c > ma:
                five = rows[max(0, i - 4) : i + 1]
                if c >= max(r["close"] for r in five) - 1e-9:
                    # swing low before R1 ~ min of prior 10 days
                    prior = rows[max(0, i - 10) : i]
                    sim.r1_swing_low = min(r["low"] for r in prior) if prior else lo
                    buy_tier(sim, "R1", i, row)
                    sim.r1_at = i

        # fake right
        if sim.r1_at is not None and "R2" not in sim.fired and sim.r1_swing_low is not None:
            if lo < sim.r1_swing_low * 0.99:
                sim.r2_frozen = True

        # R2: 15 sessions after R1 without new low, or after -30%
        if (
            sim.r1_at is not None
            and not sim.r2_frozen
            and "R2" not in sim.fired
            and (i - sim.r1_at >= 15 or c <= 0.70 * H)
        ):
            buy_tier(sim, "R2", i, row)

        v = nav(sim, c)
        sim.nav_series.append({"date": row["date"], "nav": v, "px": c})
        if sim.first_buy_i is not None:
            sim.peak_nav = max(sim.peak_nav, v)
            dd = (v / sim.peak_nav) - 1
            sim.max_dd = min(sim.max_dd, dd)

    # mark checkpoints
    def at(date: str):
        for p in sim.nav_series:
            if p["date"] >= date:
                return p
        return sim.nav_series[-1] if sim.nav_series else None

    last = sim.nav_series[-1]
    deployed = sum(b["usd"] for b in sim.buys)
    return {
        "label": label,
        "variant": variant,
        "buys": sim.buys,
        "fired": sorted(sim.fired.keys(), key=lambda t: list(DEFAULT_PCT).index(t)),
        "deployed": round(deployed, 2),
        "deployed_pct": round(100 * deployed / sim.B, 2),
        "cash_left": round(sim.cash, 2),
        "shares": round(sim.shares, 4),
        "avg_cost": round((deployed / sim.shares) if sim.shares else 0, 2),
        "max_dd_pct": round(100 * sim.max_dd, 2),
        "end_nav": round(last["nav"], 2),
        "end_date": last["date"],
        "end_ret_pct": round(100 * (last["nav"] / sim.B - 1), 2),
        "checkpoints": {},
    }


def main():
    rows = fetch_qqq()
    print(f"loaded {len(rows)} QQQ bars {rows[0]['date']} -> {rows[-1]['date']}")

    def run_full(start_date, end_date, variant, marks, h_mode="rolling60"):
        """h_mode: rolling60 = live bot rule; cycle_peak = freeze H at episode peak (手册「明确波段高点」)."""
        sim = Sim(variant=variant)
        idxs = [i for i, r in enumerate(rows) if start_date <= r["date"] <= end_date]
        # seed cycle peak with 60d lookback high at episode start
        if h_mode == "cycle_peak" and idxs:
            sim.H = rolling_high(rows, idxs[0], 60)
        for i in idxs:
            row = rows[i]
            if h_mode == "cycle_peak":
                sim.H = max(sim.H or 0.0, row["high"])
            else:
                sim.H = rolling_high(rows, i, 60)
            H = sim.H
            c = row["close"]
            lo = row["low"]
            for tier, m in MULT.items():
                if c <= m * H:
                    if tier in ("T5", "T6", "T7") and not sim.has_minus22:
                        continue
                    buy_tier(sim, tier, i, row)
                    if tier == "T4" or c <= 0.78 * H:
                        sim.has_minus22 = True
            if lo <= 0.75 * H:
                sim.has_minus22 = True
                buy_tier(sim, "T4", i, row)
            if sim.has_minus22 and "R1" not in sim.fired and not sim.r2_frozen:
                ma = sma(rows, i, 20)
                if ma is not None and c > ma:
                    five = rows[max(0, i - 4) : i + 1]
                    if c >= max(r["close"] for r in five) - 1e-9:
                        prior = rows[max(0, i - 10) : i]
                        sim.r1_swing_low = min(r["low"] for r in prior) if prior else lo
                        buy_tier(sim, "R1", i, row)
                        sim.r1_at = i
            if sim.r1_at is not None and "R2" not in sim.fired and sim.r1_swing_low is not None:
                if lo < sim.r1_swing_low * 0.99:
                    sim.r2_frozen = True
            if (
                sim.r1_at is not None
                and not sim.r2_frozen
                and "R2" not in sim.fired
                and (i - sim.r1_at >= 15 or c <= 0.70 * H)
            ):
                buy_tier(sim, "R2", i, row)
            v = nav(sim, c)
            sim.nav_series.append({"date": row["date"], "nav": round(v, 2), "px": round(c, 2)})
            if sim.first_buy_i is not None:
                if sim.first_buy_i == i:
                    sim.peak_nav = v
                else:
                    sim.peak_nav = max(sim.peak_nav, v)
                sim.max_dd = min(sim.max_dd, v / sim.peak_nav - 1)
        deployed = sum(b["usd"] for b in sim.buys)

        def cp(date):
            for p in sim.nav_series:
                if p["date"] >= date:
                    return {
                        "date": p["date"],
                        "nav": p["nav"],
                        "ret_pct": round(100 * (p["nav"] / sim.B - 1), 2),
                        "px": p["px"],
                    }
            if not sim.nav_series:
                return None
            p = sim.nav_series[-1]
            return {
                "date": p["date"],
                "nav": p["nav"],
                "ret_pct": round(100 * (p["nav"] / sim.B - 1), 2),
                "px": p["px"],
            }

        step = 5 if h_mode == "cycle_peak" or (end_date[:4] <= "2023") else 3
        return {
            "variant": variant,
            "h_mode": h_mode,
            "fired": sorted(sim.fired.keys(), key=lambda t: list(DEFAULT_PCT).index(t)),
            "buys": sim.buys,
            "deployed": round(deployed, 2),
            "deployed_pct": round(100 * deployed / sim.B, 2),
            "cash_left": round(sim.cash, 2),
            "avg_cost": round((deployed / sim.shares) if sim.shares else 0, 2),
            "shares": round(sim.shares, 4),
            "max_dd_pct": round(100 * sim.max_dd, 2),
            "end": cp(end_date),
            "marks": {m: cp(m) for m in marks},
            "nav_series": sim.nav_series[::step],
        }

    # (start, end, label, marks, h_mode)
    episodes = [
        (
            "2020-02-01",
            "2020-12-31",
            "2020 COVID 急跌急涨（Feb → 年末）",
            ["2020-03-23", "2020-04-01", "2020-06-08", "2020-09-02", "2020-12-31"],
            "rolling60",
        ),
        (
            "2022-01-03",
            "2023-12-29",
            "2022 熊市 + 2023 修复（滚动 60 日 H）",
            ["2022-06-16", "2022-10-14", "2022-12-30", "2023-06-30", "2023-12-29"],
            "rolling60",
        ),
        (
            "2022-01-03",
            "2023-12-29",
            "2022 熊市 + 2023 修复（波段高点 H，测危机袋）",
            ["2022-06-16", "2022-10-14", "2022-12-30", "2023-06-30", "2023-12-29"],
            "cycle_peak",
        ),
        (
            "2025-02-01",
            "2025-12-31",
            "2025 年春抄底（Feb 高点 → 年末）",
            ["2025-04-07", "2025-04-08", "2025-06-10", "2025-10-29", "2025-12-31"],
            "rolling60",
        ),
        (
            "2026-01-15",
            "2026-09-08",
            "2026 年春浅回撤（Jan 高点 → 至今）",
            ["2026-03-30", "2026-04-02", "2026-06-02", "2026-09-08"],
            "rolling60",
        ),
    ]

    payload = {"asof": rows[-1]["date"], "B": 100000, "model": "equity-proxy", "episodes": []}
    for start, end, label, marks, h_mode in episodes:
        ep = {
            "label": label,
            "start": start,
            "end": end,
            "marks": marks,
            "h_mode": h_mode,
            "variants": {},
        }
        for variant in ("default", "vBoost"):
            ep["variants"][variant] = run_full(start, end, variant, marks, h_mode)
        d = ep["variants"]["default"]
        v = ep["variants"]["vBoost"]
        print("\n===", label, "===")
        for variant in ("default", "vBoost"):
            r = ep["variants"][variant]
            print(
                f"{variant:8} fired={','.join(r['fired']) or '-'} "
                f"deployed={r['deployed_pct']}% (${r['deployed']:,.0f}) "
                f"avg={r['avg_cost']} end={r['end']['nav']:,.0f} ({r['end']['ret_pct']:+.1f}%) "
                f"maxDD={r['max_dd_pct']:.1f}%"
            )
            print("  buys:", "; ".join(f"{b['tier']}@{b['date']} ${b['usd']:.0f}/{b['px']:.2f}" for b in r["buys"]))
        ep["delta"] = {
            "end_ret_pp": round(v["end"]["ret_pct"] - d["end"]["ret_pct"], 2),
            "max_dd_pp": round(v["max_dd_pct"] - d["max_dd_pct"], 2),
            "deployed_pp": round(v["deployed_pct"] - d["deployed_pct"], 2),
            "marks": {
                m: {
                    "ret_pp": round(v["marks"][m]["ret_pct"] - d["marks"][m]["ret_pct"], 2)
                    if v["marks"].get(m) and d["marks"].get(m)
                    else None
                }
                for m in marks
            },
        }
        payload["episodes"].append(ep)
        print("DELTA", ep["delta"])

    out_path = r"d:\code\qqq-dip-playbook\vboost-compare.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
