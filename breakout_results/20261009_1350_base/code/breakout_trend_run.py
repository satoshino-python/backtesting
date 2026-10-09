"""
スイングブレイクアウト・トレンドフォロー戦略（docs/BREAKOUT_TREND_SPEC.md）を全ペアで検証する。

仕様書 12章の手順をまとめて実行する。
  1. 初期値（BASE）で全期間を検証
  2. パラメータを1つずつ変えた感度分析（SENSITIVITY）
  3. ウォークフォワード: 前半（IS）で各パラメータの値を選び、組み合わせた設定を後半（OOS）で確かめる
  4. コスト2倍
売買の判定・約定は4時間足だけで行う（breakout_trend.simulate）。1分足は4時間足を作るのと、チャートに使う。

出力: RESULTS_ROOT/<日時>_<RUN_LABEL>/
  trades_base.csv / trades_all_variants.csv / summary.csv / yearly.csv / by_reason.csv / sensitivity.csv /
  walkforward.csv / cost.csv / equity.csv / mfe_mae.csv / *.png（matplotlib） / dow_trade_chart_<ペア>_base.html /
  report_data.json（レポート用） / config.json / code/
"""
import contextlib
import dataclasses
import io
import json
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from breakout_trend import BreakoutParams, simulate, summary, equity_curve, equity_max_dd_pct, REASONS

# ===== 変更するパラメータはここだけ =====
PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY", "SP500")
START, END = "2021-01-01", "2025-12-31"     # 取引日（NY 17:00 区切り）。SP500 はデータのある 2024-01 から
IS_END = "2023-12-31"                       # ウォークフォワードの前半の終わり（エントリー日で分ける）
WARMUP_MONTHS = 1                           # 助走期間（ATR・スイング用）
RISK_PCT = 0.01                             # 1トレードのリスク（資金曲線用）
CASH = 10_000                               # チャートの損益表示用（1R = CASH × RISK_PCT）
SLIPPAGE = {}                               # ペア → スリッページ（価格単位、1回の約定ごと）。無ければ 0
SPREAD_OVERRIDE = {}                        # ペア → スプレッド（価格単位）。無ければデータの平均
BASE = BreakoutParams()
SENSITIVITY = {
    "fractal_n": (2, 3, 5),
    "atr_period": (14, 20),
    "sl_atr": (1.0, 1.25, 1.5, 1.75, 2.0),
    "time_bars": (4, 6, 9, 12),
    "time_atr": (0.5, 0.75, 1.0, 1.25, 1.5),
    "be_atr": (1.0, 1.25, 1.5, 1.75, 2.0),
    "trail_atr": (1.5, 2.0, 2.5, 3.0),
    "trail_swing": (True, False),
    "max_bars": (30, 60, 90),
    "atr_mode": ("fixed", "update"),
}
COST_MULTS = (1.0, 2.0)
RESULTS_ROOT = Path("breakout_results")
RUN_LABEL = "base"
WORKERS = 3                                 # 1ペアの1分足の読み込みに 1〜2GB 使う
MAKE_CHARTS = True                          # トレード付きのダウ理論チャート（ペアごとに約2.7MB）
# =====================================


def variants():
    """(名前, パラメータ) の一覧。base と、SENSITIVITY の各値（初期値と同じものは除く）"""
    out = [("base", BASE)]
    for key, values in SENSITIVITY.items():
        for v in values:
            if v != getattr(BASE, key):
                out.append((f"{key}={v}", dataclasses.replace(BASE, **{key: v})))
    return out


def pair_start(pair):
    return "2024-01-01" if pair in ("SP500", "US500") else START


def price_decimals_for(pair):
    name = pair.upper()
    if name in ("SP500", "US500"):
        return 2
    return 3 if name.endswith("JPY") else 5


def localize(index):
    """NY 現地時刻（タイムゾーンなし）→ タイムゾーン付き。チャートの取引履歴用"""
    return pd.DatetimeIndex(index).tz_localize("America/New_York", ambiguous=np.ones(len(index), bool),
                                               nonexistent="shift_forward")


def chart_trades(tr):
    """breakout_trend のトレード一覧 → make_chart() が読める形（backtesting.py の _trades と同じ列名）"""
    risk = CASH * RISK_PCT
    return pd.DataFrame({
        "EntryTime": localize(tr["EntryTime"]), "ExitTime": localize(tr["ExitTime"]),
        "Size": tr["Dir"].to_numpy(), "EntryPrice": tr["EntryPrice"].to_numpy(),
        "ExitPrice": tr["ExitPrice"].to_numpy(), "SL": tr["FinalStop"].to_numpy(), "TP": np.nan,
        "PnL": tr["R"].to_numpy() * risk, "Reason": tr["Reason"].to_numpy(),
    })


def load_pair(pair, out_dir, make_charts):
    """1ペアの1分足を読み、4時間足とスプレッド（価格単位）を返す。チャートもここで作る"""
    from main_4H_fixedSL import load_gmo_click_1min_data, resample_to_signal_bars
    from dow_swing_chart import make_chart

    start, end = pd.Timestamp(pair_start(pair)), pd.Timestamp(END)
    lo = (start.to_period("M") - WARMUP_MONTHS)
    hi = end.to_period("M") + 1
    log = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dec = price_decimals_for(pair)
        df, rel_spread = load_gmo_click_1min_data(
            f"histData/{pair}", price_side="mid", price_decimals=dec,
            month_range=(lo.year * 100 + lo.month, hi.year * 100 + hi.month), spread_period=(start, end))
        bars = resample_to_signal_bars(df, 17, 4)
        from main_4H_fixedSL import signal_bar_trading_day
        day = signal_bar_trading_day(bars.index, 17)
        mean_price = bars.loc[(day >= start) & (day <= end), "Close"].mean()
        spread = SPREAD_OVERRIDE.get(pair, rel_spread * mean_price)
        if make_charts:
            tr = simulate(bars, BASE, spread=spread, slippage=SLIPPAGE.get(pair, 0.0), symbol=pair,
                          start=start, end=end)
            from main_4H_fixedSL import get_trading_day_label
            days = get_trading_day_label(df.index, 17)
            first, last = days[days >= start][0], days[days <= end][-1]
            make_chart(df, out_dir / f"dow_trade_chart_{pair}_base.html",
                       f"{pair} {first:%Y-%m-%d} ~ {last:%Y-%m-%d} / トレード: スイングブレイクアウト（初期値）",
                       first, last, trades=chart_trades(tr), weekly_trend=None, mode="base",
                       risk=CASH * RISK_PCT, entry_window=BASE.fractal_n, entry_atr=BASE.atr_period,
                       price_decimals=dec)
    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return pair, bars, float(spread), float(rel_spread), time.time() - t0


def run_variant(bars_by_pair, spreads, params, cost_mult=1.0):
    frames = []
    for pair, bars in bars_by_pair.items():
        frames.append(simulate(bars, params, spread=spreads[pair], slippage=SLIPPAGE.get(pair, 0.0),
                               cost_mult=cost_mult, symbol=pair, start=pair_start(pair), end=END))
    tr = pd.concat(frames, ignore_index=True)
    return tr.sort_values(["ExitTime", "Symbol"], kind="stable").reset_index(drop=True)


def entry_day(tr):
    """エントリーした足の取引日（NY 17:00 区切り）"""
    from main_4H_fixedSL import signal_bar_trading_day
    return pd.Series(signal_bar_trading_day(pd.DatetimeIndex(tr["EntryTime"]), 17), index=tr.index)


def split(tr):
    """エントリー日で前半（IS）と後半（OOS）に分ける"""
    is_mask = entry_day(tr) <= pd.Timestamp(IS_END)
    return tr[is_mask], tr[~is_mask]


def stat_row(name, tr, **extra):
    s = summary(tr)
    eq = equity_curve(tr, RISK_PCT)
    s["max_dd_pct"] = equity_max_dd_pct(eq)
    s["return_pct"] = float((eq.iloc[-1] - 1) * 100) if len(eq) else 0.0
    return dict(name=name, **extra, **s)


def mfe_reach(tr, a=3.0, b=6.0):
    reached_a = (tr["MFE_ATR"] >= a).sum()
    reached_b = (tr["MFE_ATR"] >= b).sum()
    return int(reached_a), int(reached_b), float(reached_b / reached_a * 100) if reached_a else np.nan


def plots(out_dir, tr, eq_all, eq_pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(eq_all.index, (eq_all - 1) * 100, lw=1.4, color="#2a5bd7", label="All")
    ax.set_title(f"Equity (fixed {RISK_PCT:.1%} risk, compounding)")
    ax.set_ylabel("%")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_all.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for pair, eq in eq_pairs.items():
        ax.plot(eq.index, (eq - 1) * 100, lw=1.1, label=pair)
    ax.set_title(f"Equity by symbol (fixed {RISK_PCT:.1%} risk)")
    ax.set_ylabel("%")
    ax.grid(alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "equity_by_pair.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tr["R"], bins=np.arange(np.floor(tr["R"].min()), np.ceil(tr["R"].max()) + 0.25, 0.25),
            color="#2a5bd7", alpha=.85)
    ax.axvline(0, color="k", lw=.8)
    ax.set_title(f"R distribution (n={len(tr)}, mean {tr['R'].mean():+.3f}R)")
    ax.set_xlabel("R")
    fig.tight_layout()
    fig.savefig(out_dir / "r_hist.png", dpi=110)
    plt.close(fig)

    n3, n6, pct = mfe_reach(tr)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tr["MFE_ATR"], bins=np.arange(0, tr["MFE_ATR"].max() + 0.5, 0.25), color="#18865a", alpha=.85)
    for x in (3, 6):
        ax.axvline(x, color="k", lw=.8, ls="--")
    ax.set_title(f"MFE (ATR) / reached +3ATR: {n3}, of which +6ATR: {n6} ({pct:.1f}%)")
    ax.set_xlabel("MFE (ATR)")
    fig.tight_layout()
    fig.savefig(out_dir / "mfe_hist.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(tr["MAE_ATR"], bins=np.arange(0, tr["MAE_ATR"].max() + 0.25, 0.1), color="#c8413a", alpha=.85)
    ax.axvline(BASE.sl_atr, color="k", lw=.8, ls="--")
    ax.set_title("MAE (ATR) / dashed = initial stop")
    ax.set_xlabel("MAE (ATR)")
    fig.tight_layout()
    fig.savefig(out_dir / "mae_hist.png", dpi=110)
    plt.close(fig)


def save_run_info(out_dir, spreads, rel_spreads):
    info = dict(
        run_label=RUN_LABEL, started_at=datetime.now().isoformat(timespec="seconds"),
        pairs=list(spreads), start=START, end=END, is_end=IS_END, warmup_months=WARMUP_MONTHS,
        risk_pct=RISK_PCT, base=dataclasses.asdict(BASE),
        sensitivity={k: list(v) for k, v in SENSITIVITY.items()}, cost_mults=list(COST_MULTS),
        spread_price=spreads, spread_relative=rel_spreads, slippage=SLIPPAGE,
    )
    (out_dir / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code = out_dir / "code"
    code.mkdir(exist_ok=True)
    for name in ("breakout_trend.py", "breakout_trend_run.py"):
        shutil.copy2(Path(__file__).with_name(name), code / name)


def main():
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out_dir.mkdir(parents=True, exist_ok=False)
    pairs = [p for p in (sys.argv[1:] or PAIRS)]
    t0 = time.time()
    bars_by_pair, spreads, rel_spreads = {}, {}, {}
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(load_pair, p, out_dir, MAKE_CHARTS) for p in pairs]
        for f in futs:
            pair, bars, spread, rel, sec = f.result()
            bars_by_pair[pair], spreads[pair], rel_spreads[pair] = bars, spread, rel
            print(f"{pair}: 4時間足 {len(bars):,} 本 / スプレッド {spread:.6g}（{rel * 1e4:.2f}bp） / {sec:.0f}秒", flush=True)
    save_run_info(out_dir, spreads, rel_spreads)

    # 1. 初期値
    base = run_variant(bars_by_pair, spreads, BASE)
    base.to_csv(out_dir / "trades_base.csv", index=False, encoding="utf-8-sig")
    rows = [stat_row("全体", base)] + [stat_row(p, base[base.Symbol == p]) for p in pairs]
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    year = entry_day(base).dt.year
    yearly = base.assign(Year=year).pivot_table(index="Year", columns="Symbol", values="R", aggfunc="sum")
    yearly["合計"] = yearly.sum(axis=1)
    yearly.to_csv(out_dir / "yearly.csv", encoding="utf-8-sig")
    yearly_stats = pd.DataFrame([stat_row(int(y), base[year == y]) for y in sorted(year.unique())])
    yearly_stats.to_csv(out_dir / "yearly_stats.csv", index=False, encoding="utf-8-sig")
    by_reason = (base.groupby("Reason")["R"].agg(count="count", mean="mean", sum="sum")
                 .reindex([v for v in REASONS.values() if v in set(base.Reason)]))
    by_reason["share_pct"] = by_reason["count"] / len(base) * 100
    by_reason.to_csv(out_dir / "by_reason.csv", encoding="utf-8-sig")
    by_side = pd.DataFrame([stat_row(s, base[base.Side == s]) for s in ("買い", "売り")])

    eq_all = equity_curve(base, RISK_PCT)
    eq_pairs = {p: equity_curve(base[base.Symbol == p], RISK_PCT) for p in pairs}
    pd.DataFrame({"equity_all": eq_all}).to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
    n3, n6, pct = mfe_reach(base)
    plots(out_dir, base, eq_all, eq_pairs)

    # 2. 感度分析（全期間・IS・OOS）
    sens_rows, all_tr = [], [base.assign(Variant="base")]
    results = {}
    for name, params in variants():
        tr = base if name == "base" else run_variant(bars_by_pair, spreads, params)
        results[name] = tr
        if name != "base":
            all_tr.append(tr.assign(Variant=name))
        ins, oos = split(tr)
        key = name.split("=")[0] if "=" in name else ""
        value = name.split("=")[1] if "=" in name else ""
        r = stat_row(name, tr, param=key, value=value)
        r.update({f"is_{k}": v for k, v in summary(ins).items() if k in ("trades", "expectancy_R", "total_R", "pf")})
        r.update({f"oos_{k}": v for k, v in summary(oos).items() if k in ("trades", "expectancy_R", "total_R", "pf")})
        r["mfe3_to_6_pct"] = mfe_reach(tr)[2]
        sens_rows.append(r)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(out_dir / "sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_tr, ignore_index=True).to_csv(out_dir / "trades_all_variants.csv", index=False,
                                                encoding="utf-8-sig")

    # 3. ウォークフォワード: IS の期待値が最も高い値を、パラメータごとに選ぶ
    chosen = {}
    base_is = sens.loc[sens.name == "base", "is_expectancy_R"].iloc[0]
    for key in SENSITIVITY:
        cand = sens[(sens.param == key)][["value", "is_expectancy_R"]].values.tolist()
        cand.append([str(getattr(BASE, key)), base_is])
        best = max(cand, key=lambda x: x[1])
        chosen[key] = best[0]

    def cast(key, v):
        t = type(getattr(BASE, key))
        return (v == "True") if t is bool else t(v)
    wf_params = dataclasses.replace(BASE, **{k: cast(k, v) for k, v in chosen.items()})
    wf_tr = run_variant(bars_by_pair, spreads, wf_params)
    wf_rows = []
    for label, tr in (("初期値", base), ("前半で選んだ設定", wf_tr)):
        ins, oos = split(tr)
        wf_rows.append(stat_row(label, ins, period=f"前半 {START[:7]}〜{IS_END[:7]}"))
        oos_start = (pd.Timestamp(IS_END) + pd.Timedelta(days=1)).strftime("%Y-%m")
        wf_rows.append(stat_row(label, oos, period=f"後半 {oos_start}〜{END[:7]}"))
    wf = pd.DataFrame(wf_rows)
    wf.to_csv(out_dir / "walkforward.csv", index=False, encoding="utf-8-sig")
    wf_tr.to_csv(out_dir / "trades_walkforward.csv", index=False, encoding="utf-8-sig")

    # 4. コスト2倍
    cost_rows = []
    for m in COST_MULTS:
        tr = base if m == 1.0 else run_variant(bars_by_pair, spreads, BASE, cost_mult=m)
        cost_rows.append(stat_row(f"コスト×{m:g}", tr, cost_mult=m))
        ins, oos = split(tr)
        cost_rows[-1].update(is_expectancy_R=summary(ins)["expectancy_R"],
                             oos_expectancy_R=summary(oos)["expectancy_R"])
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(out_dir / "cost.csv", index=False, encoding="utf-8-sig")

    # レポート用のデータ
    def records(df):
        return json.loads(df.to_json(orient="records", force_ascii=False))
    eq_daily = eq_all.groupby(eq_all.index.normalize()).last()
    cum_r = base.groupby(base["ExitTime"].dt.normalize())["R"].sum().cumsum()
    report = dict(
        created=datetime.now().isoformat(timespec="minutes"), out_dir=str(out_dir),
        start=START, end=END, is_end=IS_END, risk_pct=RISK_PCT, pairs=pairs, base=dataclasses.asdict(BASE),
        spreads={p: dict(price=spreads[p], bp=rel_spreads[p] * 1e4) for p in pairs},
        summary=records(summary_df), yearly_stats=records(yearly_stats),
        yearly={str(y): {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in row.items()}
                for y, row in yearly.iterrows()},
        by_reason=records(by_reason.reset_index()), by_side=records(by_side),
        mfe=dict(n3=n3, n6=n6, pct=pct, values=base["MFE_ATR"].round(2).tolist()),
        mae=dict(values=base["MAE_ATR"].round(2).tolist(),
                 win=base.loc[base.R > 0, "MAE_ATR"].round(2).tolist()),
        r_values=base["R"].round(3).tolist(),
        equity=dict(t=[d.strftime("%Y-%m-%d") for d in eq_daily.index],
                    pct=((eq_daily - 1) * 100).round(2).tolist()),
        cum_r=dict(t=[d.strftime("%Y-%m-%d") for d in cum_r.index], r=cum_r.round(2).tolist()),
        cum_r_pairs={p: base[base.Symbol == p]["R"].cumsum().round(2).tolist() for p in pairs},
        sensitivity=records(sens), walkforward=records(wf),
        wf_params={k: str(v) for k, v in chosen.items()}, cost=records(cost),
        charts=[f"dow_trade_chart_{p}_base.html" for p in pairs] if MAKE_CHARTS else [],
    )
    (out_dir / "report_data.json").write_text(json.dumps(report, ensure_ascii=False, default=str),
                                              encoding="utf-8")

    pd.set_option("display.width", 200)
    cols = ["name", "trades", "win_rate", "expectancy_R", "total_R", "pf", "payoff", "max_dd_R",
            "max_losing_streak", "max_dd_pct", "return_pct"]
    print("\n== 初期値 ==")
    print(summary_df[cols].round(3).to_string(index=False))
    print("\n== 決済理由 ==")
    print(by_reason.round(3).to_string())
    print(f"\nMFE: +3ATR 到達 {n3} 件のうち +6ATR 到達 {n6} 件（{pct:.1f}%）")
    print("\n== 感度分析（全期間 / IS / OOS の期待値） ==")
    print(sens[["name", "trades", "expectancy_R", "total_R", "pf", "is_expectancy_R", "oos_expectancy_R"]]
          .round(3).to_string(index=False))
    print("\n== ウォークフォワード ==", chosen)
    print(wf[["name", "period", "trades", "expectancy_R", "total_R", "pf"]].round(3).to_string(index=False))
    print("\n== コスト ==")
    print(cost[["name", "trades", "expectancy_R", "total_R", "pf", "is_expectancy_R", "oos_expectancy_R"]]
          .round(3).to_string(index=False))
    print(f"\n結果: {out_dir}（{time.time() - t0:.0f}秒）")


if __name__ == "__main__":
    main()
