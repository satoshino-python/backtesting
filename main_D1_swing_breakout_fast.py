"""
日足スイングブレイクアウト（シンプル版）を fast_engine で全ペア検証する。

【エントリー】日足（NY 17:00 区切り）の前後 ENTRY_WINDOW 本（既定20）の Swing High に買い逆指値 / Swing Low に売り逆指値
              （共通ルール。一度抜かれたラインは次のスイング確定までエントリーしない）。フィルターなし。
【決済】TP・建値ストップなし。日足の前後 EXIT_WINDOW 本（既定10）のスイングを抜けたら決済
  - 買い: 直近の Swing Low にストップ（抜けたら決済）、売り: 直近の Swing High にストップ
  - 新しい決済スイングが確定したら、有利な方向にだけストップを移す（トレーリング）
  - 直近の決済スイングがエントリー価格の反対側に無いときは発注しない
【枚数】当初の決済ラインまでの距離で 1R（初期資金 × risk_pct）になる枚数（複利なし）
実体は fast_engine の exit_mode="dow_swing"（main_4H_dow_swingExit.py と同じ売買ルール）で、判定足を日足、フィルターなしにしたもの。
結果は fast_results/<日時>_<RUN_LABEL>/ に保存する。
"""
import io, json, shutil, dataclasses, contextlib, time, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as M
from main_4H_fixedSL import (Config, load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
                             map_signals_to_1min, get_trading_day_label, add_extra_stats)
from main_4H_dow_swingExit import compute_exit_swings
from fast_engine import run_fast
from swap_costs import load_interest_table, usdjpy_close_series, apply_interest, cumulative_interest_curve
from dow_swing_chart import make_chart

ENTRY_WINDOW = 20
import os
EXIT_WINDOW = int(os.environ.get("EXIT_WINDOW", 10))
BASE_CFG = Config(start_date="2021-01-01", end_date="2025-12-31", signal_hours=24,
                  window=ENTRY_WINDOW, atr_period=14, risk_pct=0.02)
DATA_ROOT = Path("histData")
PAIRS = None
WARMUP_MONTHS = 4                  # 前後20本の判定に必要な助走期間（日足 約60本）
RESULTS_ROOT = Path("fast_results")
RUN_LABEL = f"D1_swing{ENTRY_WINDOW}_exit{EXIT_WINDOW}"
WORKERS = 4
INTEREST_CSV = {"XAUUSD": Path("swap_data/gold_spot_interest.csv")}
INTEREST_FX_PAIR = "USDJPY"
# 枚数は整数のため、1R=200ドルでは SP500 は枚数0（注文なし）、金は1オンス単位に粗くなる。資金を倍率分だけ大きくして枚数の丸めを細かくする（Rは同じ意味）
CASH_SCALE = {"SP500": 100, "XAUUSD": 100}
MAKE_CHARTS = True


def month_range(cfg):
    p0 = pd.Timestamp(cfg.start_date).to_period("M") - WARMUP_MONTHS
    p1 = pd.Timestamp(cfg.end_date).to_period("M") + 1
    return p0.year * 100 + p0.month, p1.year * 100 + p1.month


def run_pair(pair, cfg, out_dir):
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        mr = month_range(cfg)
        df_1min, spread = load_gmo_click_1min_data(
            cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
            month_range=mr, spread_period=(start_ts, end_ts), session_start_hour=cfg.session_start_hour)
        bars = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
        sig = compute_signals(bars, window=cfg.window, atr_period=cfg.atr_period, price_decimals=cfg.price_decimals)
        df = map_signals_to_1min(df_1min, sig, cfg.session_start_hour, cfg.signal_hours)
        df = df.join(compute_exit_swings(df_1min, EXIT_WINDOW, cfg.session_start_hour, cfg.signal_hours))
        df["DowTrend"] = 0.0
        day = get_trading_day_label(df.index, cfg.session_start_hour)
        df = df.loc[(day >= start_ts) & (day <= end_ts)]
        if df.empty:
            raise ValueError("指定した検証期間にデータがありません")
        rate = spread / 2
        commission = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)
        t_load = time.perf_counter() - t0
        stats = add_extra_stats(run_fast(df, cash=cfg.cash, commission=commission, margin=cfg.margin,
                                         price_decimals=cfg.price_decimals, risk_pct=cfg.risk_pct,
                                         exit_mode="dow_swing"))
        t_bt = time.perf_counter() - t0 - t_load
        print(stats)
        risk = cfg.cash * cfg.risk_pct
        trades = stats["_trades"].copy()
        equity = stats["_equity_curve"]["Equity"]
        if pair in INTEREST_CSV:
            fx, _ = load_gmo_click_1min_data(DATA_ROOT / INTEREST_FX_PAIR, price_side="mid",
                                             price_decimals=M.price_decimals_for(INTEREST_FX_PAIR), month_range=mr,
                                             spread_period=(start_ts, end_ts), session_start_hour=cfg.session_start_hour)
            trades = apply_interest(trades, load_interest_table(INTEREST_CSV[pair]), usdjpy_close_series(fx), cfg.session_start_hour)
            equity = equity + cumulative_interest_curve(trades, equity.index)
            stats["Equity Final [$]"] = float(equity.iloc[-1])
            stats["Return [%]"] = (equity.iloc[-1] / cfg.cash - 1) * 100
            stats["Max. Drawdown [%]"] = float(((equity - equity.cummax()) / equity.cummax()).min() * 100)
        trades.insert(0, "Pair", pair)
        trades["R"] = trades["PnL"] / risk
        r_curve = (equity - cfg.cash) / risk
        chart_path = None
        if MAKE_CHARTS:
            d0, d1 = get_trading_day_label(df.index, cfg.session_start_hour)[[0, -1]]
            chart_path = out_dir / f"dow_trade_chart_{pair}_{d0:%Y%m%d}-{d1:%Y%m%d}.html"
            make_chart(df_1min, chart_path, f"{pair} {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d} / 日足 前後{ENTRY_WINDOW}ブレイクで買い売り・前後{EXIT_WINDOW}ブレイクで決済",
                       d0, d1, trades=trades.drop(columns=["Pair", "R"]), weekly_trend=None, mode="dow_swing", risk=risk,
                       entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
                       session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours)
    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    return dict(pair=pair, stats=stats[[k for k in stats.index if not str(k).startswith("_")]], trades=trades,
                r_curve_hourly=r_curve.resample("1h").last().dropna(), max_dd_r=M.max_drawdown(r_curve),
                chart_path=chart_path, seconds=time.perf_counter() - t0)


def main():
    pairs = list(PAIRS) if PAIRS else sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(dict(
        entry_window=ENTRY_WINDOW, exit_window=EXIT_WINDOW, pairs=pairs,
        base_cfg={k: str(v) for k, v in dataclasses.asdict(BASE_CFG).items()}), ensure_ascii=False, indent=2), encoding="utf-8")
    code = out_dir / "code"; code.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for n in (Path(__file__).name, "fast_engine.py", "main_4H_fixedSL.py", "main_4H_dow_swingExit.py", "swap_costs.py"):
        shutil.copy2(here / n, code / n)
    results = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(pairs))) as ex:
        futs = {ex.submit(run_pair, p, dataclasses.replace(BASE_CFG, data_path=DATA_ROOT / p, cash=BASE_CFG.cash * CASH_SCALE.get(p, 1),
                          price_decimals=M.price_decimals_for(p)), out_dir): p for p in pairs}
        for f in as_completed(futs):
            p = futs[f]
            try:
                results[p] = f.result()
            except Exception as e:
                print(f"❌ {p}: {e!r}"); continue
            print(f"✅ {p}: {results[p]['stats']['# Trades']} トレード / {results[p]['trades']['R'].sum():+.1f}R ({results[p]['seconds']:.0f}秒)")
    results = {p: results[p] for p in pairs if p in results}
    rows = {}
    for p, r in results.items():
        row = M.r_stats(r["trades"].sort_values(["ExitTime", "EntryTime"], kind="stable")["R"])
        row["最大DD [R]"] = r["max_dd_r"]; row["収益率 [%]"] = r["stats"]["Return [%]"]; row["最大DD [%]"] = r["stats"]["Max. Drawdown [%]"]
        rows[p] = row
    allt = pd.concat([r["trades"] for r in results.values()], ignore_index=True).sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    tot = M.r_stats(allt["R"])
    curve = pd.concat({p: r["r_curve_hourly"] for p, r in results.items()}, axis=1, sort=True).sort_index().ffill().fillna(0).sum(axis=1)
    tot["最大DD [R]"] = M.max_drawdown(curve); tot["収益率 [%]"] = tot["合計R"] * BASE_CFG.risk_pct / len(results) * 100; tot["最大DD [%]"] = float("nan")
    rows["合計"] = tot
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary[["トレード数", "最大連敗"]] = summary[["トレード数", "最大連敗"]].astype(int)
    summary.index.name = "通貨ペア"
    M.print_table("通貨ペア別 / 総合成績（R倍数）", summary)
    allt["Year"] = pd.to_datetime(allt["ExitTime"]).dt.year
    yr = allt.pivot_table(index="Year", columns="Pair", values="R", aggfunc="sum", fill_value=0.0); yr["合計"] = yr.sum(axis=1)
    yn = allt.pivot_table(index="Year", columns="Pair", values="R", aggfunc="count", fill_value=0); yn["合計"] = yn.sum(axis=1)
    M.print_table("年別 合計R", yr)
    summary.round(4).to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    pd.concat({"合計R": yr, "トレード数": yn}, axis=1).round(4).to_csv(out_dir / "yearly.csv", encoding="utf-8-sig")
    allt.drop(columns="Year").to_csv(out_dir / "trades_all.csv", index=False, encoding="utf-8-sig")
    curve.rename("CumR").to_csv(out_dir / "equity_R_combined.csv", encoding="utf-8-sig")
    print(f"\n💾 {out_dir.resolve()}")


if __name__ == "__main__":
    main()
