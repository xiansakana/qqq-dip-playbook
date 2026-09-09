#!/usr/bin/env python3
"""Default vs V-boost with equity+LEAP, dual-currency wallets (CNY-first equity).

Cash modes:
  usd_only  — $100k USD (previous canvas baseline)
  half      — $50k USD + ¥350k CNY (B=$100k @ FX=7)
  cny_heavy — $20k USD + ¥560k CNY

Equity: spend CNY first into a QQQ-linked CNY proxy (synthetic 159509);
remainder USD buys QQQ. Options: USD only. TP to equity → QQQ (USD proceeds).
"""
from __future__ import annotations

import json
import math
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

UA = {"User-Agent": "Mozilla/5.0"}
R = 0.04
FX = 7.0

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

CASH_MODES = {
    "usd_only": {"cash_usd": 100_000.0, "cash_cny": 0.0},
    "half": {"cash_usd": 50_000.0, "cash_cny": 350_000.0},
    "cny_heavy": {"cash_usd": 20_000.0, "cash_cny": 560_000.0},
}


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


def attach_cny_proxy(rows, px_159509: dict, px_513100: dict):
    """Use real 159509 when listed; else 513100 (older Nasdaq A-share ETF); else None."""
    out = []
    for r in rows:
        d = r["date"]
        src = None
        px = None
        if d in px_159509:
            px, src = px_159509[d], "159509"
        elif d in px_513100:
            px, src = px_513100[d], "513100"
        out.append({**r, "proxy": px, "proxy_src": src})
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
    cash_mode: str
    option_policy: str = "playbook"
    B: float = 100_000.0
    cash_usd: float = 100_000.0
    cash_cny: float = 0.0
    shares_qqq: float = 0.0
    shares_proxy: float = 0.0
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
    spent_usd: float = 0.0
    spent_cny: float = 0.0
    last_proxy: float | None = None
    proxy_src_used: set = field(default_factory=set)


def nav_of(sim: Sim, S: float, proxy: float | None, asof: date, vxn: float | None) -> float:
    opt = sum(lot.contracts * 100.0 * lot.mark(S, asof, vxn) for lot in sim.calls if lot.contracts > 0)
    px = proxy if proxy is not None and proxy > 0 else sim.last_proxy
    proxy_mv = sim.shares_proxy * px / FX if px and px > 0 else 0.0
    return (
        sim.cash_usd
        + sim.cash_cny / FX
        + sim.shares_qqq * S
        + proxy_mv
        + opt
    )


def buy_equity(sim: Sim, usd: float, S: float, proxy: float | None, tier: str, d: str,
               proxy_src: str | None = None) -> float:
    """CNY-first into onshore proxy when priced; else USD QQQ. Returns USD-eq spent."""
    need = max(0.0, usd)
    if need <= 1:
        return 0.0
    spent_eq = 0.0
    label = proxy_src or "159509"

    cny_avail_usd = sim.cash_cny / FX if FX > 0 else 0.0
    cny_usd = min(need, cny_avail_usd)
    if cny_usd > 1 and proxy is not None and proxy > 0:
        cny_amt = cny_usd * FX
        sim.cash_cny -= cny_amt
        sim.shares_proxy += cny_amt / proxy
        sim.spent_cny += cny_amt
        spent_eq += cny_usd
        need -= cny_usd
        sim.buys.append({
            "tier": tier, "kind": f"equity-{label}", "date": d,
            "px": round(proxy, 4), "usd": round(cny_usd, 2), "cny": round(cny_amt, 2),
        })
    elif cny_usd > 1 and (proxy is None or proxy <= 0):
        # No onshore print that day — spill CNY budget to USD QQQ if possible.
        pass

    usd_buy = min(need, sim.cash_usd)
    if usd_buy > 1 and S > 0:
        sim.cash_usd -= usd_buy
        sim.shares_qqq += usd_buy / S
        sim.spent_usd += usd_buy
        spent_eq += usd_buy
        sim.buys.append({
            "tier": tier, "kind": "equity-QQQ", "date": d,
            "px": round(S, 2), "usd": round(usd_buy, 2), "cny": 0.0,
        })
    return spent_eq


def buy_call(sim, usd, S, asof, vxn, kind, tier, from_r1=False) -> float:
    usd = min(usd, sim.cash_usd)
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
        return 0.0
    contracts = math.floor(usd / (prem * 100.0))
    if contracts < 1:
        return 0.0
    spend = contracts * prem * 100.0
    sim.cash_usd -= spend
    sim.spent_usd += spend
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
        "contracts": contracts, "usd": round(spend, 2), "cny": 0.0, "vxn": vxn,
    })
    return spend


def deploy_tier(sim: Sim, tier: str, row: dict, asof: date):
    if tier in sim.fired:
        return
    budget = round(sim.B * tier_pct(tier, sim.variant), 2)
    wallet = sim.cash_usd + sim.cash_cny / FX
    if budget <= 0 or wallet < 1:
        return
    S, proxy, vxn = row["close"], row.get("proxy"), row.get("vxn")
    if proxy is not None and proxy > 0:
        sim.last_proxy = proxy
        if row.get("proxy_src"):
            sim.proxy_src_used.add(row["proxy_src"])
    blocked = tier in VXN_MIN and (vxn is None or vxn < VXN_MIN[tier])
    right_only = sim.option_policy == "right_only"
    spent = 0.0
    src = row.get("proxy_src")

    def eq(usd):
        return buy_equity(sim, usd, S, proxy, tier, asof.isoformat(), proxy_src=src)

    def call(usd, kind, from_r1=False):
        got = buy_call(sim, usd, S, asof, vxn, kind, tier, from_r1=from_r1)
        # If USD insufficient for options, spill remainder to equity (CNY-first).
        if got + 1 < usd:
            got += eq(usd - got)
        return got

    if tier in ("T1", "T5", "R2"):
        spent += eq(budget)
    elif tier == "T2":
        if blocked or right_only:
            spent += eq(budget)
        else:
            spent += eq(budget * 0.75)
            spent += call(budget * 0.25, "medium")
    elif tier == "T3":
        if blocked or right_only:
            spent += eq(budget)
        else:
            spent += eq(budget * 0.55)
            spent += call(budget * 0.35, "medium")
            spent += call(budget * 0.10, "deep")
    elif tier == "T4":
        if right_only:
            spent += eq(budget)
        else:
            spent += eq(budget * 0.80)
            spent += call(budget * 0.20, "medium")
    elif tier in ("T6", "T7"):
        if right_only:
            spent += eq(budget)
        else:
            spent += eq(budget * 0.70)
            spent += call(budget * 0.30, "shallow")
    elif tier == "R1":
        if blocked:
            spent += eq(budget)
        elif right_only:
            spent += eq(budget * 0.50)
            spent += call(budget * 0.50, "medium", from_r1=True)
        else:
            spent += eq(budget * 0.80)
            spent += call(budget * 0.20, "shallow", from_r1=True)
    sim.fired[tier] = {"date": asof.isoformat(), "budget": budget, "spent": round(spent, 2)}


def apply_take_profit(sim: Sim, S: float, asof: date, vxn: float | None):
    keep = []
    for lot in sim.calls:
        if lot.contracts <= 0:
            continue
        if asof >= lot.expiry:
            intrinsic = max(S - lot.strike, 0.0)
            sim.cash_usd += lot.contracts * 100.0 * intrinsic
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
                sim.shares_qqq += proceeds / S
            else:
                sim.cash_usd += proceeds
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
            sim.cash_usd += lot.contracts * 100.0 * px
            sim.tp_events.append({
                "date": asof.isoformat(), "reason": "fakeRight-clear-R1-opt",
                "kind": lot.kind, "strike": lot.strike, "contracts": lot.contracts,
                "to": "cash", "usd": round(lot.contracts * 100.0 * px, 2),
            })
        else:
            keep.append(lot)
    sim.calls = keep


def run_full(rows, start_date, end_date, variant, marks, h_mode="rolling60",
             option_policy="playbook", cash_mode="half"):
    wallets = CASH_MODES[cash_mode]
    sim = Sim(
        variant=variant, cash_mode=cash_mode, option_policy=option_policy,
        cash_usd=wallets["cash_usd"], cash_cny=wallets["cash_cny"],
        peak_nav=100_000.0,
    )
    idxs = [i for i, r in enumerate(rows) if start_date <= r["date"] <= end_date]
    if not idxs:
        raise SystemExit(f"no rows {start_date}..{end_date}")
    if h_mode == "cycle_peak":
        sim.H = rolling_high(rows, idxs[0], 60)

    for i in idxs:
        row = rows[i]
        asof = parse_d(row["date"])
        S, lo, proxy, vxn = row["close"], row["low"], row.get("proxy"), row.get("vxn")
        if proxy is not None and proxy > 0:
            sim.last_proxy = proxy
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

        v = nav_of(sim, S, proxy, asof, vxn)
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
    eq_qqq = sim.shares_qqq * last["close"]
    last_px = last.get("proxy") or sim.last_proxy or 0.0
    eq_proxy = sim.shares_proxy * last_px / FX if last_px else 0.0

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

    n_proxy = sum(1 for b in sim.buys if b["kind"].startswith("equity-") and b["kind"] != "equity-QQQ")
    n_qqq = sum(1 for b in sim.buys if b["kind"] == "equity-QQQ")
    n_call = sum(1 for b in sim.buys if b["kind"].startswith("call-"))
    proxy_kinds = sorted({b["kind"] for b in sim.buys if b["kind"].startswith("equity-") and b["kind"] != "equity-QQQ"})

    return {
        "variant": variant, "cash_mode": cash_mode, "h_mode": h_mode,
        "option_policy": option_policy,
        "model": "equity+synthetic-leap+real-cny-etf",
        "proxy_sources": sorted(sim.proxy_src_used),
        "proxy_buy_kinds": proxy_kinds,
        "fired": sorted(sim.fired.keys(), key=lambda t: list(DEFAULT_PCT).index(t)),
        "buys": sim.buys, "tp_count": len(sim.tp_events),
        "deployed": round(deployed, 2), "deployed_pct": round(100 * deployed / sim.B, 2),
        "spent_usd": round(sim.spent_usd, 2),
        "spent_cny": round(sim.spent_cny, 2),
        "cash_usd_left": round(sim.cash_usd, 2),
        "cash_cny_left": round(sim.cash_cny, 2),
        "equity_qqq_mv": round(eq_qqq, 2),
        "equity_proxy_mv": round(eq_proxy, 2),
        "option_mv": round(opt_mv, 2),
        "buy_counts": {"proxy": n_proxy, "qqq": n_qqq, "call": n_call},
        "max_dd_pct": round(100 * sim.max_dd, 2),
        "end": cp(end_date), "marks": {m: cp(m) for m in marks},
    }


def main():
    print("fetching QQQ + VXN + 159509 + 513100…")
    qqq = fetch_yahoo("QQQ")
    vxn = fetch_yahoo("^VXN")
    etf159 = {r["date"]: r["close"] for r in fetch_yahoo("159509.SZ", start="2023-01-01")}
    etf513 = {r["date"]: r["close"] for r in fetch_yahoo("513100.SS")}
    rows = attach_cny_proxy(align(qqq, vxn), etf159, etf513)
    n159 = sum(1 for r in rows if r.get("proxy_src") == "159509")
    n513 = sum(1 for r in rows if r.get("proxy_src") == "513100")
    print(
        f"aligned {len(rows)} bars {rows[0]['date']} -> {rows[-1]['date']} "
        f"(proxy days 159509={n159}, 513100={n513})"
    )

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
        "asof": rows[-1]["date"], "B": 100000, "fx": FX,
        "model": "equity+synthetic-leap+real-cny-etf",
        "proxy": "159509.SZ when listed (from 2023-07-19); else 513100.SS",
        "cash_modes": CASH_MODES,
        "episodes": [],
    }

    for start, end, label, marks, h_mode in episodes:
        ep = {
            "label": label, "start": start, "end": end, "marks": marks,
            "h_mode": h_mode, "modes": {},
        }
        print(f"\n=== {label} [{h_mode}] ===")
        for cash_mode in ("usd_only", "half", "cny_heavy"):
            ep["modes"][cash_mode] = {}
            for variant in ("default", "vBoost"):
                r = run_full(rows, start, end, variant, marks, h_mode, cash_mode=cash_mode)
                ep["modes"][cash_mode][variant] = r
                print(
                    f"{cash_mode:10} {variant:8} fired={','.join(r['fired']) or '-'} "
                    f"dep={r['deployed_pct']}% end={r['end']['ret_pct']:+.1f}% "
                    f"dd={r['max_dd_pct']:.1f}% "
                    f"qqq={r['equity_qqq_mv']:.0f} proxy={r['equity_proxy_mv']:.0f} "
                    f"opt={r['option_mv']:.0f} src={r['proxy_sources']} "
                    f"buys={r['buy_counts']} spentUSD={r['spent_usd']:.0f} spentCNY={r['spent_cny']:.0f}"
                )
            d = ep["modes"][cash_mode]["default"]
            v = ep["modes"][cash_mode]["vBoost"]
            ep["modes"][cash_mode]["delta"] = {
                "end_ret_pp": round(v["end"]["ret_pct"] - d["end"]["ret_pct"], 2),
                "max_dd_pp": round(v["max_dd_pct"] - d["max_dd_pct"], 2),
                "deployed_pp": round(v["deployed_pct"] - d["deployed_pct"], 2),
            }
        u = ep["modes"]["usd_only"]["default"]
        h = ep["modes"]["half"]["default"]
        c = ep["modes"]["cny_heavy"]["default"]
        ep["fx_delta_default"] = {
            "half_vs_usd_end_pp": round(h["end"]["ret_pct"] - u["end"]["ret_pct"], 2),
            "cny_vs_usd_end_pp": round(c["end"]["ret_pct"] - u["end"]["ret_pct"], 2),
            "half_vs_usd_dd_pp": round(h["max_dd_pct"] - u["max_dd_pct"], 2),
            "cny_vs_usd_dd_pp": round(c["max_dd_pct"] - u["max_dd_pct"], 2),
        }
        print("FXΔ default", ep["fx_delta_default"], "proxy_src", h.get("proxy_sources"))
        payload["episodes"].append(ep)

    out = r"d:\code\qqq-dip-playbook\vboost-compare-fx.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
