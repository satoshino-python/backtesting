"""
移動平均クロスのドテン売買を、高速エンジン（fast_engine.run_fast(exit_mode="ma_cross")）で全ペア検証する。

ルール（判定足は SIGNAL_HOURS で選ぶ。既定は4時間足、環境変数 SIGNAL_HOURS=24 で日足。NY 17:00 区切り）:
  - 短期SMA（MA_FAST=75）が長期SMA（MA_SLOW=200）を終値ベースで上抜いた足が確定したら、次の足の始値で買う
  - 下抜いた足が確定したら、次の足の始値で買いを決済して売りにドテン（その後に上抜いたら買いにドテン）
  - SL / TP なし。常にどちらかのポジションを持つ（検証期間の最初のクロスまでは様子見）
  - 枚数: 1R（初期資金 × risk_pct）÷（ATR × sl_atr_multiplier）。ATR の 1.5倍逆行で 1R の損失になる量。
    他の戦略（SL 1.5 ATR）と同じ大きさのポジションにして、R をそろえるため。ATR は判定足の EMA(18)（確定済みの1本前）
  - 期末に残ったポジションは最後の終値で決済して集計に入れる（CLOSE_AT_END）
  - スプレッド（平均の半分を往復それぞれ）とスワップを反映する

結果は RESULTS_ROOT/<日時>_<RUN_LABEL>/ に保存する（summary.csv・yearly.csv・trades_*.csv・チャート・report.html）。
"""
import contextlib
import os
import dataclasses
import html
import io
import json
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
from main_4H_fixedSL import (Config, load_gmo_click_1min_data, resample_to_signal_bars, calculate_atr_ema,
                             get_signal_bar_label, get_trading_day_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for, r_stats, max_drawdown
from fast_engine import run_fast
from swap_costs import build_swap_model, SWAP_SOURCES
from dow_swing_chart import make_chart

# ===== 変更するパラメータはここだけ =====
SIGNAL_HOURS = int(os.environ.get("SIGNAL_HOURS", "4"))   # 判定足（4: 4時間足 / 24: 日足）
BASE_CFG = Config(start_date="2021-01-01", end_date="2025-12-31", sl_atr_multiplier=1.5, signal_hours=SIGNAL_HOURS)
TF = {24: "日足"}.get(SIGNAL_HOURS, f"{SIGNAL_HOURS}時間足")
TF_SHORT = {24: "D1"}.get(SIGNAL_HOURS, f"{SIGNAL_HOURS}H")
MA_FAST = 75
MA_SLOW = 200
DATA_ROOT = Path("histData")
PAIRS = ("AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPAUD", "GBPCHF",
         "GBPJPY", "GBPUSD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY")
# MA_SLOW 本の助走が要るので、既定の1ヶ月から延ばす（4時間足 200本 ≒ 1.6ヶ月、日足 200本 ≒ 10ヶ月）
WARMUP_MONTHS = 11 if SIGNAL_HOURS >= 24 else 3
CLOSE_AT_END = True
RESULTS_ROOT = Path("fast_results")
RUN_LABEL = f"FX15ペア_MAクロスドテン_{MA_FAST}x{MA_SLOW}_{TF_SHORT}"
WORKERS = 4
# 比較に並べる実行（同じ15ペア・期間）。無ければ比較なし
if SIGNAL_HOURS == 4:
    BASELINE_DIR = Path("fast_results/20261014_0500_FX15ペア_スワップあり_fixed_sl1.5_tp2.5")
    BASELINE_NAME = "スイングブレイク基準版（SL1.5/TP2.5）"
else:
    BASELINE_DIR = Path("fast_results/20261016_1207_FX15ペア_MAクロスドテン_75x200_4H")
    BASELINE_NAME = "4時間足版（同じMAクロス）"

multi.WARMUP_MONTHS = WARMUP_MONTHS  # zip_month_range() が参照する（子プロセスでも import 時に設定される）


def cross_target(h4_close, fast, slow):
    """
    判定足ごとの「次の足で持つ向き」（1 / -1、クロス前は NaN）と、その足でクロスしたかを返す。
    MA は終値の単純移動平均。短期 = 長期の足は向きを変えない（前の向きのまま）。
    """
    ma_f = h4_close.rolling(fast).mean()
    ma_s = h4_close.rolling(slow).mean()
    state = np.sign(ma_f - ma_s).replace(0, np.nan).ffill()
    state[ma_s.isna()] = np.nan
    prev = state.shift(1)
    cross = state.notna() & prev.notna() & (state != prev)
    target = state.where(cross).ffill()
    return target, cross, ma_f, ma_s


def run_pair(pair, cfg, out_dir):
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df_1min, spread = load_gmo_click_1min_data(
            cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
            month_range=zip_month_range(cfg), spread_period=(s, e), session_start_hour=cfg.session_start_hour)
        h4 = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
        target, cross, _, _ = cross_target(h4["Close"], MA_FAST, MA_SLOW)
        atr = pd.Series(calculate_atr_ema(h4["High"], h4["Low"], h4["Close"], cfg.atr_period, cfg.price_decimals),
                        index=h4.index)
        # 判定足の終値で決まった値は次の足から使う（1本シフト = 次の足の最初の1分足の始値で約定）
        sig = pd.DataFrame({"TargetPos": target, "ATR": atr}).shift(1)
        mapped = sig.reindex(get_signal_bar_label(df_1min.index, cfg.session_start_hour, cfg.signal_hours))
        day = get_trading_day_label(df_1min.index, cfg.session_start_hour)
        inp = np.asarray((day >= s) & (day <= e))
        df = df_1min.loc[inp, ["Open", "High", "Low", "Close", "Volume"]].copy()
        tp = mapped["TargetPos"].to_numpy(float)[inp]
        # 検証期間に入ってから最初のクロスまでは様子見（期間前のクロスで建てたポジションは持たない）
        tz = np.nan_to_num(tp)
        changed = np.flatnonzero(tz[1:] != tz[:-1]) + 1
        first = int(changed[0]) if len(changed) else len(tz)
        tp[:first] = 0
        df["TargetPos"] = tp
        df["SignalATR"] = mapped["ATR"].to_numpy(float)[inp]

        rate = spread / 2
        risk = cfg.cash * cfg.risk_pct

        def commission_func(order_size, price):
            return round(abs(order_size) * price * rate, cfg.price_decimals)

        swap = build_swap_model(cfg.data_path, cfg.start_date, cfg.end_date, cfg.session_start_hour)
        stats = run_fast(df, cash=cfg.cash, commission=commission_func, margin=cfg.margin,
                         price_decimals=cfg.price_decimals, risk_pct=cfg.risk_pct,
                         sl_atr_multiplier=cfg.sl_atr_multiplier, exit_mode="ma_cross",
                         close_at_end=CLOSE_AT_END, swap=swap)
        print(stats)
        trades = stats["_trades"].copy()
        if "PnL_Gross" not in trades:
            trades["PnL_Gross"] = trades["PnL"]
            trades["Interest"] = 0.0
        trades.insert(0, "Pair", pair)
        trades["R"] = trades["PnL"] / risk
        trades["R_Gross"] = trades["PnL_Gross"] / risk
        trades["Side"] = np.where(trades["Size"] > 0, "買い", "売り")
        trades["HoldDays"] = (pd.to_datetime(trades["ExitTime"]) - pd.to_datetime(trades["EntryTime"])).dt.total_seconds() / 86400
        equity = stats["_equity_curve"]["Equity"]
        r_curve = (equity - cfg.cash) / risk

        fd, ld = day[inp][[0, -1]]
        chart_name = f"dow_trade_chart_{pair}_{fd:%Y%m%d}-{ld:%Y%m%d}_ma_cross.html"
        make_chart(
            df_1min, out_dir / chart_name,
            f"{pair} {fd:%Y-%m-%d} ~ {ld:%Y-%m-%d} / トレード: {TF_SHORT} SMA{MA_FAST}×SMA{MA_SLOW} クロスでドテン",
            fd, ld, trades=stats["_trades"], weekly_trend=None, mode="ma_cross", risk=risk,
            entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
            session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours,
            filter_state=pd.Series(tp, index=df.index),
            filter_label=f"{TF_SHORT} SMA{MA_FAST} と SMA{MA_SLOW} の位置で決まる保有方向（1=買い / -1=売り / 0=最初のクロス前）",
            ma_periods=(MA_FAST, MA_SLOW))
    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    public = stats[[k for k in stats.index if not str(k).startswith("_")]]
    return dict(pair=pair, stats=public, trades=trades, r_curve_hourly=r_curve.resample("1h").last().dropna(),
                max_dd_r=max_drawdown(r_curve), chart=chart_name, spread=spread,
                seconds=time.perf_counter() - t0)


def summarize(results, out_dir):
    rows = {}
    for pair, res in results.items():
        t = res["trades"].sort_values(["ExitTime", "EntryTime"], kind="stable")
        row = r_stats(t["R"])
        row["合計R（反映前）"] = float(t["R_Gross"].sum())
        row["PF（反映前）"] = r_stats(t["R_Gross"])["プロフィットファクター"]
        row["スワップR"] = float((t["R"] - t["R_Gross"]).sum())
        row["平均保有日数"] = float(t["HoldDays"].mean()) if len(t) else float("nan")
        row["最大DD [R]"] = res["max_dd_r"]
        rows[pair] = row
    all_trades = pd.concat([r["trades"] for r in results.values()], ignore_index=True)
    all_trades = all_trades.sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    total = r_stats(all_trades["R"])
    total["合計R（反映前）"] = float(all_trades["R_Gross"].sum())
    total["PF（反映前）"] = r_stats(all_trades["R_Gross"])["プロフィットファクター"]
    total["スワップR"] = float((all_trades["R"] - all_trades["R_Gross"]).sum())
    total["平均保有日数"] = float(all_trades["HoldDays"].mean())
    combined = (pd.concat({p: r["r_curve_hourly"] for p, r in results.items()}, axis=1, sort=True)
                .sort_index().ffill().fillna(0).sum(axis=1))
    total["最大DD [R]"] = max_drawdown(combined)
    rows["合計"] = total
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "通貨ペア"

    all_trades["Year"] = pd.to_datetime(all_trades["ExitTime"]).dt.year
    yearly = all_trades.pivot_table(index="Year", columns="Pair", values="R", aggfunc="sum", fill_value=0.0)
    yearly["合計"] = yearly.sum(axis=1)
    yearly["合計（反映前）"] = all_trades.groupby("Year")["R_Gross"].sum()
    yearly["取引数"] = all_trades.groupby("Year")["R"].count()

    summary.round(4).to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    yearly.round(4).to_csv(out_dir / "yearly.csv", encoding="utf-8-sig")
    all_trades.drop(columns="Year").to_csv(out_dir / "trades_all.csv", index=False, encoding="utf-8-sig")
    for pair, res in results.items():
        res["trades"].to_csv(out_dir / f"trades_{pair}.csv", index=False, encoding="utf-8-sig")
    combined.rename("CumR").to_csv(out_dir / "equity_R_combined.csv", encoding="utf-8-sig")
    pd.DataFrame({p: r["stats"] for p, r in results.items()}).to_csv(out_dir / "stats_by_pair.csv", encoding="utf-8-sig")
    return summary, yearly, all_trades, combined


# ===================== レポート（report.html） =====================
_CSS = """
:root { --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --accent-soft:#e3eafb; --pos:#18865a; --neg:#c8413a; --gold:#9a7412;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gold:#e0b84a; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gold:#e0b84a; color-scheme:dark; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 var(--f-head); -webkit-text-size-adjust:100%; }
.wrap { max-width:860px; margin:0 auto; padding-block:20px 48px; padding-left:max(16px,env(safe-area-inset-left,0px)); padding-right:max(16px,env(safe-area-inset-right,0px)); display:grid; gap:26px; }
.wrap > * { min-width:0; }
h1 { font-size:1.45rem; line-height:1.3; margin:0; font-weight:700; text-wrap:balance; }
h2 { font-size:1.05rem; margin:0 0 10px; font-weight:700; text-wrap:balance; }
p { margin:0; } .sub { color:var(--muted); font-size:.85rem; margin-top:6px; }
.num { font-family:var(--f-num); font-variant-numeric:tabular-nums; } .pos { color:var(--pos); } .neg { color:var(--neg); } .muted { color:var(--muted); }
.cmp { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.cmp div { background:var(--surface); padding:10px 12px; min-width:0; }
.cmp .l { font-size:.7rem; color:var(--muted); letter-spacing:.04em; }
.cmp .v { font-family:var(--f-num); font-size:1.2rem; font-weight:600; white-space:nowrap; }
.cmp .was { font-family:var(--f-num); font-size:.72rem; color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }
.card p { font-size:.88rem; color:var(--muted); } .card p + p { margin-top:6px; } .card b { color:var(--fg); font-weight:500; }
svg { width:100%; height:auto; display:block; }
.gl { stroke:var(--line); stroke-width:1; } .zl { stroke:var(--muted); stroke-width:1; stroke-dasharray:2 3; } .ax { fill:var(--muted); font:11px var(--f-num); }
.l-n { fill:none; stroke:var(--accent); stroke-width:2; } .l-g { fill:none; stroke:var(--gold); stroke-width:1.5; stroke-dasharray:4 3; }
.l-b { fill:none; stroke:var(--muted); stroke-width:1.5; stroke-dasharray:1 3; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.78rem; color:var(--muted); margin-top:6px; }
.legend i { display:inline-block; width:18px; height:0; border-top:2px solid var(--accent); vertical-align:middle; margin-right:6px; }
.legend i.g { border-top:2px dashed var(--gold); } .legend i.b { border-top:2px dotted var(--muted); }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--line); border-radius:12px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:.82rem; font-family:var(--f-num); }
th,td { padding:7px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--line); }
th { font-weight:500; color:var(--muted); font-size:.72rem; font-family:var(--f-head); }
th:first-child,td:first-child { text-align:left; } tr:last-child td { border-bottom:0; } tr.tot td { background:var(--accent-soft); font-weight:600; }
.link { display:block; text-decoration:none; color:var(--fg); background:var(--surface); border:1.5px solid var(--accent); border-radius:12px; padding:14px; }
.link:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.link b { display:block; } .link span { font-size:.82rem; color:var(--muted); }
ul { margin:0; padding-left:1.2em; display:grid; gap:6px; font-size:.88rem; color:var(--muted); } li b { color:var(--fg); font-weight:500; }
code { font-family:var(--f-num); font-size:.78rem; background:var(--accent-soft); padding:1px 5px; border-radius:4px; }
.links{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px} .links .link{padding:10px 12px}
"""


def _sgn(v, fmt="{:+.1f}"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '<span class="muted">-</span>'
    cls = "pos" if v > 0 else "neg" if v < 0 else ""
    return f'<span class="{cls}">{fmt.format(v)}</span>'


def _f(v, fmt):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)


def _svg_curves(series, labels):
    """series: [(pd.Series(index=時刻, 値=累積R), css クラス)]。共通の時間軸に折れ線を描く"""
    W, H, L, R, T, B = 640, 250, 44, 628, 12, 222
    t0 = min(s.index[0] for s, _ in series)
    t1 = max(s.index[-1] for s, _ in series)
    lo = min(0.0, min(float(s.min()) for s, _ in series))
    hi = max(0.0, max(float(s.max()) for s, _ in series))
    span = hi - lo or 1.0
    step = [v for v in (5, 10, 20, 25, 50, 100, 200, 250, 500) if span / v <= 8][0]
    lo_t, hi_t = np.floor(lo / step) * step, np.ceil(hi / step) * step

    def x(t):
        return L + (R - L) * (t - t0).total_seconds() / max((t1 - t0).total_seconds(), 1)

    def y(v):
        return B - (B - T) * (v - lo_t) / (hi_t - lo_t)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(labels)}">']
    v = lo_t
    while v <= hi_t + 1e-9:
        out.append(f'<line x1="{L}" x2="{R}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="{"zl" if v == 0 else "gl"}"/>'
                   f'<text x="{L - 6}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v:g}</text>')
        v += step
    for yr in range(t0.year + 1, t1.year + 1):
        tt = pd.Timestamp(f"{yr}-01-01", tz=t0.tz)
        out.append(f'<text x="{x(tt):.1f}" y="{H - 10}" class="ax" text-anchor="middle">{yr}</text>')
    for s, cls in series:
        s = s.resample("1D").last().dropna() if len(s) > 2000 else s
        pts = " L".join(f"{x(t):.1f},{y(v):.1f}" for t, v in s.items())
        out.append(f'<path d="M{pts}" class="{cls}"/>')
    out.append("</svg>")
    return "".join(out)


def _baseline():
    p = BASELINE_DIR / "summary.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, index_col=0)
    if "合計" not in s.index:
        return None
    row = s.loc["合計"]
    t = pd.read_csv(BASELINE_DIR / "trades_all.csv")
    risk = BASE_CFG.cash * BASE_CFG.risk_pct
    gross = t["PnL_Gross"] / risk if "PnL_Gross" in t else t["R"]
    t = t.assign(ExitTime=pd.to_datetime(t["ExitTime"], utc=True).dt.tz_convert("America/New_York"))
    curve = t.sort_values("ExitTime").set_index("ExitTime")["R"].cumsum()
    return dict(total=float(row["合計R"]), pf=float(row["プロフィットファクター"]), dd=float(row["最大DD [R]"]),
                n=int(row["トレード数"]), wr=float(row["勝率 [%]"]), gross=float(gross.sum()), curve=curve,
                name=BASELINE_DIR.name)


def build_report(out_dir, results, summary, yearly, all_trades, combined, run_started):
    risk = BASE_CFG.cash * BASE_CFG.risk_pct
    tot = summary.loc["合計"]
    base = _baseline()
    t = all_trades.copy()
    t["ExitTime"] = pd.to_datetime(t["ExitTime"], utc=True).dt.tz_convert("America/New_York")
    closed_net = t.set_index("ExitTime")["R"].cumsum()
    closed_gross = t.set_index("ExitTime")["R_Gross"].cumsum()
    dd_closed_net = max_drawdown(closed_net.reset_index(drop=True))
    dd_closed_gross = max_drawdown(closed_gross.reset_index(drop=True))
    last_day = pd.Timestamp(BASE_CFG.end_date)
    side = t.groupby("Side").agg(n=("R", "size"), r=("R", "sum"), g=("R_Gross", "sum"),
                                 wr=("R", lambda s: (s > 0).mean() * 100), hold=("HoldDays", "mean"))
    n_pairs = len(results)
    best = summary.drop(index="合計")["合計R"].sort_values()

    h = [f'<title>MAクロス ドテン 15ペア</title>',
         '<meta name="claude:viewport-fit-sides" content="cover">',
         '<link rel="preconnect" href="https://fonts.googleapis.com">',
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">',
         f"<style>{_CSS}</style>", '<main class="wrap">',
         f'<header><h1>MAクロス ドテン（{TF_SHORT} SMA{MA_FAST} × SMA{MA_SLOW}）FX {n_pairs}ペア</h1>'
         f'<p class="sub">{BASE_CFG.start_date} ~ {BASE_CFG.end_date} / {TF}の終値で SMA{MA_FAST} が SMA{MA_SLOW} を上抜けたら次の足の始値で買い、'
         f'下抜けたら決済して売りにドテン / SL・TP なし / 1分足の始値で約定 / 1R = {risk:,.0f}（ATR×{BASE_CFG.sl_atr_multiplier} の逆行で 1R になる枚数）/ スプレッド・スワップ反映</p></header>']

    # 1. 主な数字
    cmp = [("合計R", _sgn(tot["合計R"]), f'反映前 {tot["合計R（反映前）"]:+.1f}' + (f' / 比較 {base["total"]:+.1f}' if base else "")),
           ("プロフィットファクター", f'{tot["プロフィットファクター"]:.3f}', f'反映前 {tot["PF（反映前）"]:.3f}' + (f' / 比較 {base["pf"]:.3f}' if base else "")),
           ("最大DD（R・含み損益込み）", f'{tot["最大DD [R]"]:.1f}', f'決済ベース {dd_closed_net:.1f}（反映前 {dd_closed_gross:.1f}）' + (f' / 比較 {base["dd"]:.1f}' if base else "")),
           ("取引数 / 勝率", f'{int(tot["トレード数"])} / {tot["勝率 [%]"]:.1f}%', (f'比較 {base["n"]} / {base["wr"]:.1f}%' if base else "")),
           ("スワップ 合計", _sgn(tot["スワップR"], "{:+.1f}R"), f'{tot["スワップR"] * risk:+,.0f} USD'),
           ("平均保有 / 平均勝ち・負け", f'{tot["平均保有日数"]:.1f}日', f'勝ち {tot["平均勝ちR"]:+.2f}R / 負け {tot["平均負けR"]:+.2f}R'),
           ]
    h.append('<section><h2>結果（スワップ反映後）</h2><div class="cmp">' + "".join(
        f'<div><div class="l">{l}</div><div class="v">{v}</div><div class="was">{w}</div></div>' for l, v, w in cmp) + "</div>")
    verdict = (f'{n_pairs}ペア合算の合計は <b>{tot["合計R"]:+.1f}R</b>（スワップ反映前 {tot["合計R（反映前）"]:+.1f}R）、'
               f'1ペアあたり平均 {tot["合計R"] / n_pairs:+.1f}R / 5年（1R = 初期資金の {BASE_CFG.risk_pct:.0%} なので、1ペア1口座で平均 {tot["合計R"] / n_pairs * BASE_CFG.risk_pct:+.0%}）。'
               f'勝率 {tot["勝率 [%]"]:.0f}% と低く、少数の大きなトレンドで稼ぐ形（平均勝ち {tot["平均勝ちR"]:+.2f}R・平均負け {tot["平均負けR"]:+.2f}R）。')
    if base:
        verdict += f' 同じ15ペア・期間の{BASELINE_NAME}は {base["total"]:+.1f}R。'
    h.append(f'<p class="sub">{verdict}</p></section>')

    # 2. 累積R
    series = [(closed_gross, "l-g"), (closed_net, "l-n")]
    legend = '<span><i></i>反映後（決済ベース）</span><span><i class="g"></i>反映前</span>'
    if base:
        series.insert(0, (base["curve"], "l-b"))
        legend += f'<span><i class="b"></i>{BASELINE_NAME}・反映後</span>'
    h.append(f'<section><h2>累積R（全{n_pairs}ペア合算）</h2><div class="card">'
             + _svg_curves(series, f"全{n_pairs}ペア合算の累積R") + f'<div class="legend">{legend}</div></div></section>')

    # 年別
    pairs = list(results)
    yr_rows = []
    for y, r in yearly.iterrows():
        yr_rows.append(f'<tr><td>{y}</td><td>{int(r["取引数"])}</td><td>{_sgn(r["合計"])}</td><td>{_sgn(r["合計（反映前）"])}</td>'
                       + "".join(f"<td>{_sgn(r.get(p, 0.0))}</td>" for p in pairs) + "</tr>")
    yr_rows.append(f'<tr class="tot"><td>合計</td><td>{int(yearly["取引数"].sum())}</td><td>{_sgn(yearly["合計"].sum())}</td>'
                   f'<td>{_sgn(yearly["合計（反映前）"].sum())}</td>' + "".join(f"<td>{_sgn(yearly[p].sum())}</td>" for p in pairs) + "</tr>")
    h.append('<section><h2>年別の合計R（決済した年）</h2><div class="scroll"><table><thead><tr><th>年</th><th>取引数</th><th>合計</th><th>反映前</th>'
             + "".join(f"<th>{p}</th>" for p in pairs) + "</tr></thead><tbody>" + "".join(yr_rows) + "</tbody></table></div></section>")

    # 買い / 売り
    sd = "".join(f'<tr><td>{s}</td><td>{int(r.n)}</td><td>{r.wr:.1f}%</td><td>{_sgn(r.r)}</td><td>{_sgn(r.g)}</td><td>{_sgn(r.r - r.g)}</td><td>{r.hold:.1f}</td></tr>'
                 for s, r in side.iterrows())
    h.append('<section><h2>買い / 売り</h2><div class="scroll"><table><thead><tr><th>方向</th><th>取引数</th><th>勝率</th><th>合計R</th><th>反映前</th><th>スワップR</th><th>平均保有日数</th></tr></thead>'
             f"<tbody>{sd}</tbody></table></div></section>")

    # ペア別
    pr = []
    for p, r in summary.iterrows():
        cls = ' class="tot"' if p == "合計" else ""
        pr.append(f'<tr{cls}><td>{p}</td><td>{int(r["トレード数"])}</td><td>{r["勝率 [%]"]:.1f}%</td><td>{_sgn(r["合計R"])}</td>'
                  f'<td>{_sgn(r["合計R（反映前）"])}</td><td>{_sgn(r["スワップR"])}</td><td>{_f(r["プロフィットファクター"], "{:.2f}")}</td>'
                  f'<td>{_f(r["平均勝ちR"], "{:+.2f}")}</td><td>{_f(r["平均負けR"], "{:+.2f}")}</td><td>{int(r["最大連敗"])}</td>'
                  f'<td>{r["最大DD [R]"]:.1f}</td><td>{r["平均保有日数"]:.1f}</td></tr>')
    h.append('<section><h2>ペア別</h2><div class="scroll"><table><thead><tr><th>ペア</th><th>取引数</th><th>勝率</th><th>合計R</th><th>反映前</th>'
             '<th>スワップR</th><th>PF</th><th>平均勝ち</th><th>平均負け</th><th>最大連敗</th><th>最大DD[R]</th><th>平均保有日数</th></tr></thead>'
             f'<tbody>{"".join(pr)}</tbody></table></div>'
             f'<p class="sub">最大DD は含み損益込み（1分足の資産曲線）。合計の行は全ペアの累積R（1時間ごと）を足した曲線の DD。</p></section>')

    # 3. チャート
    links = "".join(f'<a class="link" href="{html.escape(r["chart"])}"><b>{p}</b><span>{_sgn(float(r["trades"]["R"].sum()), "{:+.1f}R")} / {len(r["trades"])}回</span></a>'
                    for p, r in results.items())
    h.append('<section><h2>トレード付きダウ理論チャート</h2>'
             f'<p class="sub" style="margin:0 0 10px">▲買い / ▼売り / ●決済。{TF_SHORT} に切り替えると SMA{MA_FAST}・SMA{MA_SLOW} が表示されます（チャートの設定で期間を変更可）。'
             '背景は「検証で使った保有方向」（青=買い / 赤=売り）。←→キーで前後のトレードへ移動できます。</p>'
             f'<div class="links">{links}</div></section>')

    # 4. ルール・注意
    h.append('<section><h2>売買ルール・設定</h2><div class="card"><ul>'
             f'<li><b>シグナル</b>: {TF}（NY 17:00 区切り）の終値で SMA{MA_FAST} と SMA{MA_SLOW} を計算。SMA{MA_FAST} &gt; SMA{MA_SLOW} に変わった足（ゴールデンクロス）が確定したら買い、'
             f'SMA{MA_FAST} &lt; SMA{MA_SLOW} に変わった足（デッドクロス）で売り。</li>'
             f'<li><b>約定</b>: クロスした足の次の{TF}の、最初の1分足の始値（週末をまたぐ場合は週明けの始値）。保有中のポジションを同じ価格で決済して反対側を持つ（ドテン）。</li>'
             '<li><b>決済</b>: 反対のクロスだけ。SL・TP・建値ストップなし。検証期間の最初のクロスまではノーポジション（期間前のクロスで建てたポジションは持たない）。'
             f'期末（{last_day:%Y-%m-%d}）に残ったポジションは最後の終値で決済して集計に含める。</li>'
             f'<li><b>枚数</b>: 1R（{BASE_CFG.cash:,.0f} × {BASE_CFG.risk_pct:.0%} = {risk:,.0f}）÷（{TF} ATR(EMA {BASE_CFG.atr_period}) × {BASE_CFG.sl_atr_multiplier}）。'
             f'エントリー時点の{TF} ATR の {BASE_CFG.sl_atr_multiplier}倍逆行したら 1R の損失になる量（スイングブレイク基準版と同じ考え方。日足では4時間足よりポジションが小さい）。複利なし。損失は SL が無いため 1R を超えることがある。</li>'
             '<li><b>コスト</b>: 平均スプレッドの半分をエントリー・決済それぞれに課金。スワップ（GMO FXスワップシート、1万通貨あたり）を NY 17:00 をまたいだ日数分反映。</li>'
             f'<li><b>データ</b>: GMOクリック証券の1分足（仲値）。MA の助走に検証開始の {WARMUP_MONTHS} ヶ月前から読み込む。</li>'
             '<li><b>エンジン</b>: <code>fast_engine.run_fast(exit_mode="ma_cross")</code>（今回追加。backtesting.py 版は無いので答え合わせはしていない）。スクリプトは <code>main_4H_ma_cross_fast.py</code>。</li>'
             '</ul></div></section>')
    worst = ", ".join(f"{p} {v:+.1f}R" for p, v in best.head(3).items())
    top = ", ".join(f"{p} {v:+.1f}R" for p, v in best.tail(3)[::-1].items())
    h.append('<section><h2>読むときの注意</h2><div class="card"><ul>'
             f'<li><b>期間分割なし</b>: 5年を1回通しただけ。パラメータ（{MA_FAST}/{MA_SLOW}・{TF}）は一般的な値をそのまま使い、合わせ込みはしていない。</li>'
             f'<li><b>取引数が少ない</b>: 1ペアあたり平均 {tot["トレード数"] / n_pairs:.0f} 回 / 5年。少数の大きなトレンドで結果が決まるため、ペア間・年ごとのばらつきが大きい（上位: {top} / 下位: {worst}）。</li>'
             '<li><b>保有が長い</b>: 平均で数週間持ち続けるため、スワップの影響が大きい。高金利通貨の売り（例: クロス円の売り）はスワップの支払いが重い。シートの単位（1万通貨あたり）は未確認。</li>'
             '<li><b>損切りが無い</b>: 1回の損失は ATR の何倍にもなり得る（R は「ATR×1.5 の逆行 = 1R」の換算）。窓開け・急変時のリスクは R の最大DD で見る。</li>'
             '<li><b>ペアごとに別口座</b>: 15ペアを同時に持つ証拠金の取り合いは再現していない（証拠金のチェックもしない）。</li>'
             '</ul></div></section>')
    h.append(f'<p class="sub">結果フォルダ: <code>{html.escape(str(out_dir))}</code> / 実行 {run_started:%Y-%m-%d %H:%M}</p></main>')
    (out_dir / "report.html").write_text("\n".join(h), encoding="utf-8")


def main():
    pairs = [p for p in PAIRS if (DATA_ROOT / p).is_dir()]
    started = datetime.now()
    out_dir = RESULTS_ROOT / f"{started:%Y%m%d_%H%M}_{RUN_LABEL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code").mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for name in (Path(__file__).name, "fast_engine.py", "main_4H_fixedSL.py", "swap_costs.py"):
        shutil.copy2(here / name, out_dir / "code" / name)
    (out_dir / "config.json").write_text(json.dumps(dict(
        run_label=RUN_LABEL, started_at=started.isoformat(timespec="seconds"), engine="fast_engine (ma_cross)",
        ma_fast=MA_FAST, ma_slow=MA_SLOW, close_at_end=CLOSE_AT_END, warmup_months=WARMUP_MONTHS, pairs=pairs,
        swap_pairs=sorted(p for p in pairs if p in SWAP_SOURCES),
        base_cfg={k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(BASE_CFG).items()}),
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ℹ️ {len(pairs)}ペア / {BASE_CFG.start_date} ~ {BASE_CFG.end_date} / {TF_SHORT} SMA{MA_FAST}×SMA{MA_SLOW} ドテン")
    results = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(pairs))) as ex:
        futs = {ex.submit(run_pair, p, dataclasses.replace(BASE_CFG, data_path=DATA_ROOT / p,
                                                           price_decimals=price_decimals_for(p)), out_dir): p for p in pairs}
        for f in as_completed(futs):
            p = futs[f]
            try:
                results[p] = f.result()
            except Exception as e:
                print(f"❌ {p}: {e!r}")
                continue
            r = results[p]
            print(f"✅ {p}: {len(r['trades'])} 回 / {r['trades']['R'].sum():+.1f}R（反映前 {r['trades']['R_Gross'].sum():+.1f}R）"
                  f" / {r['seconds']:.0f}秒", flush=True)
    results = {p: results[p] for p in pairs if p in results}
    summary, yearly, all_trades, combined = summarize(results, out_dir)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary.round(2))
        print(yearly.round(1))
    build_report(out_dir, results, summary, yearly, all_trades, combined, started)
    print(f"💾 {out_dir}")


if __name__ == "__main__":
    main()
