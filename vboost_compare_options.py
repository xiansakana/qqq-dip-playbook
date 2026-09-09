#!/usr/bin/env python3
"""Default 40/30/30 vs V-boost with equity + synthetic LEAP Calls.

Matches live qqq-dip buy splits and VXN gates. Option marks use Black-Scholes
with wing-discounted IV from ^VXN (calibrated so 2025-04-07 QQQ 700C ≈ $2.35).
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}
R = 0.04

DEFAULT_PCT = {
    "T1": 0.06, "T2": 0.08, "T3": 0.14, "T4": 0.12,
    "T5": 0.10, "T6": 0.10, "T7": 0.10, "R1": 0.10, "R2": 0.20,
}
BAG = {
    "default": {"left": 0.40, "crisis": 0.30, "right": 0.30},
    "vBoost": {"left": 0.55, "crisis": 0.20, "right": 0.25},
}
TIER_BAG = {
    "T1": "left", "T2": "left", "T3": "left", "T4": "left",
    "T5": "crisis", "T6": "crisis", "T7": "crisis", "R1": "right", "R2": "right",
}
MULT = {"T1": 0.92, "T2": 0.88, "T3": 0.82, "T4": 0.78, "T5": 0.70, "T6": 0.60, "T7": 0.50}
VXN_MIN = {"T2": 25.0, "T3": 32.0, "R1": 25.0}


def tier_pct(tier: str, variant: str) -> float:
    base = BAG["default"][TIER_BAG[tier]]
    target = BAG[variant][TIER_BAG[tier]]
    return DEFAULT_PCT[tier] * (target / base)


def fetch_yahoo(symbol: str, start="2019-12-01", end="2026-09-10"):
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    enc = symbol.replace("^", "%5E")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{enc}"
        f"?period1={p1}&period2={p2}&interval=1d&events=history"
    )
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.load(resp)
    result = data["chart"]["result"][0]
    ts = result["timestamp"]
    q = result["indicators"]["quote"][0]
    rows = []
    for i, t in enumerate(ts):
        c = q["close"][i]
        if c is None:
            continue
        h = q["high"][i]
        lo = q["low"][i]
        rows.append({
            "date": datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d"),
            "close": float(c),
            "high": float(h) if h is not None else float(c),
            "low": float(lo) if lo is not None else float(c),
        })
    return rows


def align(qqq, vxn):
    vm = {r["date"]: r["close"] for r in vxn}
    out, last_v = [], None
    for r in qqq:
        if r["date"] in vm:
            last_v = vm[r["date"]]
        out.append({**r, "vxn": last_v})
    return out


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, r: float, sig: float) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 1 / 365:
        return max(S - K, 0.0)
    sig = max(sig, 1e-6)
    d1 = (math.log(S / K) + (r + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    d2 = d1 - sig * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def parse_d(s: str) -> date:
    return date.fromisoformat(s)


def leap_expiry(asof: date) -> date:
    for y in range(asof.year + 1, asof.year + 4):
        exp = date(y, 1, 17)
        months = (exp - asof).days / 30.44
        if 12 <= months <= 21:
            return exp
    return date(asof.year + 2, 1, 17)


def round_strike(x: float, step: float = 5.0) -> float:
    return max(step, round(x / step) * step)


def wing_iv(vxn: float | None, kind: str) -> float:
    base = (vxn / 100.0) if vxn and vxn > 0 else 0.22
    if kind == "deep":
        return max(0.12, base * 0.50)
    if kind == "shallow":
        return max(0.14, base * 0.85)
    return max(0.14, base * 0.75)


def years_to(exp: date, asof: date) -> float:
    return max(0.0, (exp - asof).days / 365.25)


def rolling_high(rows, i, lookback=60):
    window = rows[max(0, i - lookback + 1): i + 1]
    return max(r["high"] for r in window)


def sma(rows, i, n=20):
    if i + 1 < n:
        return None
    return sum(r["close"] for r in rows[i - n + 1: i + 1]) / n


@dataclass
class CallLot:
    kind: str
    strike: float
    expiry: date
    contracts: float
    cost: float
    opened: date
    tier: str
    entry_contracts: float
    sold3x: bool = False
    sold5x: bool = False
    holding_high: float = 0.0
    from_r1: bool = False

    def mark(self, S: float, asof: date, vxn: float | None) -> float:
        T = years_to(self.expiry, asof)
        px = bs_call(S, self.strike, T, R, wing_iv(vxn, self.kind))
        self.holding_high = max(self.holding_high, px)
        return px


@dataclass
class Sim:
    variant: str
    # playbook: left+R1 options; right_only: equity on left/crisis, options only from R1
    option_policy: str = "playbook"
    B: float = 100_000.0
    cash: float = 100_000.0
    shares: float = 0.0
    calls: list = field(default_factory=list)
    fired: dict = field(default_factory=dict)
    buys: list = field(default_factory=list)
    tp_events: list = field(default_factory=list)
    nav_series: list = field(default_factory=list)
    H: float | None = None
    has_minus22: bool = False
    r1_at: int | None = None
    r1_swing_low: float | None = None
    r2_frozen: bool = False
    peak_nav: float = 100_000.0
    max_dd: float = 0.0
    first_buy_i: int | None = None
    deep_spent: float = 0.0


def nav_of(sim: Sim, S: float, asof: date, vxn: float | None) -> float:
    opt = sum(lot.contracts * 100.0 * lot.mark(S, asof, vxn) for lot in sim.calls if lot.contracts > 0)
    return sim.cash + sim.shares * S + opt


def buy_equity(sim: Sim, usd: float, px: float, tier: str, d: str) -> float:
    usd = min(usd, sim.cash)
    if usd <= 1 or px <= 0:
        return 0.0
    sim.cash -= usd
    sim.shares += usd / px
    sim.buys.append({"tier": tier, "kind": "equity", "date": d, "px": round(px, 2), "usd": round(usd, 2)})
    return usd


def buy_call(sim, usd, S, asof, vxn, kind, tier, from_r1=False) -> float:
    usd = min(usd, sim.cash)
    if usd < 50 or S <= 0:
        return 0.0
    if kind == "deep":
        usd = min(usd, max(0.0, 0.08 * sim.B - sim.deep_spent))
        if usd < 50:
            return 0.0
    otm = {"medium": 0.32, "deep": 0.575, "shallow": 0.20}[kind]
    K = round_strike(S * (1.0 + otm))
    exp = leap_expiry(asof)
    prem = bs_call(S, K, years_to(exp, asof), R, wing_iv(vxn, kind))
    if prem < 0.05:
        return buy_equity(sim, usd, S, tier, asof.isoformat())
    contracts = math.floor(usd / (prem * 100.0))
    if contracts < 1:
        return buy_equity(sim, usd, S, tier, asof.isoformat())
    spend = contracts * prem * 100.0
    sim.cash -= spend
    if kind == "deep":
        sim.deep_spent += spend
    sim.calls.append(CallLot(
        kind=kind, strike=K, expiry=exp, contracts=float(contracts),
        cost=prem, opened=asof, tier=tier, entry_contracts=float(contracts),
        holding_high=prem, from_r1=from_r1,
    ))
    sim.buys.append({
        "tier": tier, "kind": f"call-{kind}", "date": asof.isoformat(),
        "strike": K, "expiry": exp.isoformat(), "prem": round(prem, 4),
        "contracts": contracts, "usd": round(spend, 2), "vxn": vxn,
    })
    return spend


def deploy_tier(sim: Sim, tier: str, row: dict, asof: date):
    if tier in sim.fired:
        return
    budget = round(sim.B * tier_pct(tier, sim.variant), 2)
    if budget <= 0 or sim.cash < 1:
        return
    S, vxn = row["close"], row.get("vxn")
    blocked = tier in VXN_MIN and (vxn is None or vxn < VXN_MIN[tier])
    right_only = sim.option_policy == "right_only"
    spent = 0.0
    if tier in ("T1", "T5", "R2"):
        spent += buy_equity(sim, budget, S, tier, asof.isoformat())
    elif tier == "T2":
        if blocked or right_only:
            spent += buy_equity(sim, budget, S, tier, asof.isoformat())
        else:
            spent += buy_equity(sim, budget * 0.75, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.25, S, asof, vxn, "medium", tier)
    elif tier == "T3":
        if blocked or right_only:
            spent += buy_equity(sim, budget, S, tier, asof.isoformat())
        else:
            spent += buy_equity(sim, budget * 0.55, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.35, S, asof, vxn, "medium", tier)
            spent += buy_call(sim, budget * 0.10, S, asof, vxn, "deep", tier)
    elif tier == "T4":
        if right_only:
            spent += buy_equity(sim, budget, S, tier, asof.isoformat())
        else:
            spent += buy_equity(sim, budget * 0.80, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.20, S, asof, vxn, "medium", tier)
    elif tier in ("T6", "T7"):
        if right_only:
            spent += buy_equity(sim, budget, S, tier, asof.isoformat())
        else:
            spent += buy_equity(sim, budget * 0.70, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.30, S, asof, vxn, "shallow", tier)
    elif tier == "R1":
        if blocked:
            spent += buy_equity(sim, budget, S, tier, asof.isoformat())
        elif right_only:
            # Deferred convexity: half of R1 into medium LEAP after right confirm.
            spent += buy_equity(sim, budget * 0.50, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.50, S, asof, vxn, "medium", tier, from_r1=True)
        else:
            spent += buy_equity(sim, budget * 0.80, S, tier, asof.isoformat())
            spent += buy_call(sim, budget * 0.20, S, asof, vxn, "shallow", tier, from_r1=True)
    sim.fired[tier] = {"date": asof.isoformat(), "budget": budget, "spent": round(spent, 2)}


def apply_take_profit(sim: Sim, S: float, asof: date, vxn: float | None):
    keep = []
    for lot in sim.calls:
        if lot.contracts <= 0:
            continue
        if asof >= lot.expiry:
            intrinsic = max(S - lot.strike, 0.0)
            sim.cash += lot.contracts * 100.0 * intrinsic
            sim.tp_events.append({
                "date": asof.isoformat(), "reason": "expiry", "kind": lot.kind,
                "strike": lot.strike, "contracts": lot.contracts, "to": "cash",
                "usd": round(lot.contracts * 100.0 * intrinsic, 2),
            })
            continue

        px = lot.mark(S, asof, vxn)
        mult = px / lot.cost if lot.cost > 0 else 0.0
        months = (asof - lot.opened).days / 30.44
        tdays = (lot.expiry - asof).days
        otm_pct = (lot.strike / S - 1.0) * 100.0 if S > 0 else 999.0
        entry = lot.entry_contracts

        def cash_out(n_contracts, to, reason):
            n = min(n_contracts, lot.contracts)
            if n <= 0:
                return
            proceeds = n * 100.0 * px
            lot.contracts -= n
            if to == "equity" and S > 0:
                sim.shares += proceeds / S
            else:
                sim.cash += proceeds
            sim.tp_events.append({
                "date": asof.isoformat(), "reason": reason, "kind": lot.kind,
                "strike": lot.strike, "contracts": round(n, 4), "to": to,
                "usd": round(proceeds, 2), "mult": round(mult, 2),
            })

        if lot.kind == "deep":
            if mult >= 3 and not lot.sold3x:
                cash_out(entry * 0.50, "equity", "deep-3x")
                lot.sold3x = True
            if mult >= 5 and not lot.sold5x:
                cash_out(entry * 0.25, "cash", "deep-5x")
                lot.sold5x = True
            if (lot.sold5x or mult >= 5) and lot.holding_high > 0 and px <= lot.holding_high * 0.6:
                cash_out(lot.contracts, "cash", "deep-trail40")
            elif months >= 6:
                cash_out(lot.contracts, "cash", "deep-6mo")
            elif tdays < 120 and otm_pct > 15:
                cash_out(lot.contracts, "cash", "deep-dte120")
        else:
            if mult >= 3 and not lot.sold3x:
                cash_out(entry / 3.0, "equity", "mid-3x")
                lot.sold3x = True
            if mult >= 5 and not lot.sold5x:
                cash_out(entry / 6.0, "equity", "mid-5x-eq")
                cash_out(entry / 6.0, "cash", "mid-5x-cash")
                lot.sold5x = True
            if lot.sold3x and lot.holding_high > 0 and px <= lot.holding_high * 0.75:
                cash_out(lot.contracts, "equity", "mid-trail")
            elif months >= 8 and mult < 3:
                cash_out(lot.contracts * 0.5, "equity", "mid-8mo")
            elif tdays < 120 and otm_pct > 15:
                cash_out(lot.contracts, "cash", "opt-dte")

        if lot.contracts > 1e-6:
            keep.append(lot)
    sim.calls = keep


def liquidate_r1_calls(sim: Sim, S: float, asof: date, vxn: float | None):
    keep = []
    for lot in sim.calls:
        if lot.from_r1 and lot.contracts > 0:
            px = lot.mark(S, asof, vxn)
            sim.cash += lot.contracts * 100.0 * px
            sim.tp_events.append({
                "date": asof.isoformat(), "reason": "fakeRight-clear-R1-opt",
                "kind": lot.kind, "strike": lot.strike, "contracts": lot.contracts,
                "to": "cash", "usd": round(lot.contracts * 100.0 * px, 2),
            })
        else:
            keep.append(lot)
    sim.calls = keep


def run_full(rows, start_date, end_date, variant, marks, h_mode="rolling60", option_policy="playbook"):
    sim = Sim(variant=variant, option_policy=option_policy)
    idxs = [i for i, r in enumerate(rows) if start_date <= r["date"] <= end_date]
    if not idxs:
        raise SystemExit(f"no rows {start_date}..{end_date}")
    if h_mode == "cycle_peak":
        sim.H = rolling_high(rows, idxs[0], 60)

    for i in idxs:
        row = rows[i]
        asof = parse_d(row["date"])
        S, lo, vxn = row["close"], row["low"], row.get("vxn")
        if h_mode == "cycle_peak":
            sim.H = max(sim.H or 0.0, row["high"])
        else:
            sim.H = rolling_high(rows, i, 60)
        H = sim.H

        apply_take_profit(sim, S, asof, vxn)

        for tier, m in MULT.items():
            if S <= m * H:
                if tier in ("T5", "T6", "T7") and not sim.has_minus22:
                    continue
                if tier not in sim.fired:
                    deploy_tier(sim, tier, row, asof)
                    if sim.first_buy_i is None:
                        sim.first_buy_i = i
                if tier == "T4" or S <= 0.78 * H:
                    sim.has_minus22 = True
        if lo <= 0.75 * H:
            sim.has_minus22 = True
            if "T4" not in sim.fired:
                deploy_tier(sim, "T4", row, asof)
                if sim.first_buy_i is None:
                    sim.first_buy_i = i

        if sim.has_minus22 and "R1" not in sim.fired and not sim.r2_frozen:
            ma = sma(rows, i, 20)
            if ma is not None and S > ma:
                five = rows[max(0, i - 4): i + 1]
                if S >= max(r["close"] for r in five) - 1e-9:
                    prior = rows[max(0, i - 10): i]
                    sim.r1_swing_low = min(r["low"] for r in prior) if prior else lo
                    deploy_tier(sim, "R1", row, asof)
                    sim.r1_at = i
                    if sim.first_buy_i is None:
                        sim.first_buy_i = i

        if sim.r1_at is not None and "R2" not in sim.fired and sim.r1_swing_low is not None:
            if lo < sim.r1_swing_low * 0.99:
                sim.r2_frozen = True
                liquidate_r1_calls(sim, S, asof, vxn)

        if (
            sim.r1_at is not None and not sim.r2_frozen and "R2" not in sim.fired
            and (i - sim.r1_at >= 15 or S <= 0.70 * H)
        ):
            deploy_tier(sim, "R2", row, asof)

        v = nav_of(sim, S, asof, vxn)
        sim.nav_series.append({"date": row["date"], "nav": round(v, 2), "px": round(S, 2)})
        if sim.first_buy_i is not None:
            if sim.first_buy_i == i:
                sim.peak_nav = v
            else:
                sim.peak_nav = max(sim.peak_nav, v)
            sim.max_dd = min(sim.max_dd, v / sim.peak_nav - 1)

    last = rows[idxs[-1]]
    asof = parse_d(last["date"])
    opt_mv = sum(
        lot.contracts * 100.0 * lot.mark(last["close"], asof, last.get("vxn"))
        for lot in sim.calls if lot.contracts > 0
    )
    deployed = sum(b["usd"] for b in sim.buys)

    def cp(d):
        for p in sim.nav_series:
            if p["date"] >= d:
                return {
                    "date": p["date"], "nav": p["nav"],
                    "ret_pct": round(100 * (p["nav"] / sim.B - 1), 2), "px": p["px"],
                }
        p = sim.nav_series[-1]
        return {
            "date": p["date"], "nav": p["nav"],
            "ret_pct": round(100 * (p["nav"] / sim.B - 1), 2), "px": p["px"],
        }

    return {
        "variant": variant, "h_mode": h_mode, "option_policy": option_policy,
        "model": "equity+synthetic-leap",
        "fired": sorted(sim.fired.keys(), key=lambda t: list(DEFAULT_PCT).index(t)),
        "buys": sim.buys, "tp_events": sim.tp_events[:50], "tp_count": len(sim.tp_events),
        "deployed": round(deployed, 2), "deployed_pct": round(100 * deployed / sim.B, 2),
        "cash_left": round(sim.cash, 2),
        "equity_mv": round(sim.shares * last["close"], 2),
        "option_mv": round(opt_mv, 2),
        "shares": round(sim.shares, 4),
        "open_calls": len([c for c in sim.calls if c.contracts > 0]),
        "deep_spent": round(sim.deep_spent, 2),
        "max_dd_pct": round(100 * sim.max_dd, 2),
        "end": cp(end_date), "marks": {m: cp(m) for m in marks},
        "nav_series": sim.nav_series[::5],
    }


def main():
    print("fetching QQQ + VXN…")
    rows = align(fetch_yahoo("QQQ"), fetch_yahoo("^VXN"))
    print(f"aligned {len(rows)} bars {rows[0]['date']} -> {rows[-1]['date']}")

    sample = next(r for r in rows if r["date"] == "2025-04-07")
    prem = bs_call(
        sample["close"], 700,
        years_to(date(2027, 1, 15), date(2025, 4, 7)), R,
        wing_iv(sample["vxn"], "deep"),
    )
    print(f"calib 2025-04-07 700C model=${prem:.2f} (target ~2.35, vxn={sample['vxn']})")

    episodes = [
        ("2020-02-01", "2020-12-31", "2020 COVID 急跌急涨",
         ["2020-03-23", "2020-04-01", "2020-06-08", "2020-09-02", "2020-12-31"], "rolling60"),
        ("2022-01-03", "2023-12-29", "2022 熊市 + 2023 修复（滚动 60 日 H）",
         ["2022-06-16", "2022-10-14", "2022-12-30", "2023-06-30", "2023-12-29"], "rolling60"),
        ("2022-01-03", "2023-12-29", "2022 熊市 + 2023 修复（波段高点 H）",
         ["2022-06-16", "2022-10-14", "2022-12-30", "2023-06-30", "2023-12-29"], "cycle_peak"),
        ("2025-02-01", "2025-12-31", "2025 年春抄底",
         ["2025-04-07", "2025-04-08", "2025-06-10", "2025-10-29", "2025-12-31"], "rolling60"),
        ("2026-01-15", "2026-09-08", "2026 年春浅回撤",
         ["2026-03-30", "2026-04-02", "2026-06-02", "2026-09-08"], "rolling60"),
    ]

    payload = {
        "asof": rows[-1]["date"], "B": 100000, "model": "equity+synthetic-leap",
        "pricing": "BSM + VXN wing haircut (deep x0.5, medium x0.75); calib 2025-04-07 700C~2.35",
        "episodes": [],
    }

    for start, end, label, marks, h_mode in episodes:
        ep = {"label": label, "start": start, "end": end, "marks": marks, "h_mode": h_mode, "variants": {}}
        print(f"\n=== {label} [{h_mode}] ===")
        for variant in ("default", "vBoost"):
            r = run_full(rows, start, end, variant, marks, h_mode)
            ep["variants"][variant] = r
            print(
                f"{variant:8} fired={','.join(r['fired']) or '-'} dep={r['deployed_pct']}% "
                f"end={r['end']['ret_pct']:+.1f}% dd={r['max_dd_pct']:.1f}% "
                f"eq={r['equity_mv']:.0f} opt={r['option_mv']:.0f} cash={r['cash_left']:.0f} tp={r['tp_count']}"
            )
        d, v = ep["variants"]["default"], ep["variants"]["vBoost"]
        ep["delta"] = {
            "end_ret_pp": round(v["end"]["ret_pct"] - d["end"]["ret_pct"], 2),
            "max_dd_pp": round(v["max_dd_pct"] - d["max_dd_pct"], 2),
            "deployed_pp": round(v["deployed_pct"] - d["deployed_pct"], 2),
            "marks": {
                m: {"ret_pp": round(v["marks"][m]["ret_pct"] - d["marks"][m]["ret_pct"], 2)
                    if v["marks"].get(m) and d["marks"].get(m) else None}
                for m in marks
            },
        }
        print("DELTA", ep["delta"])
        payload["episodes"].append(ep)

    out = r"d:\code\qqq-dip-playbook\vboost-compare-options.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
