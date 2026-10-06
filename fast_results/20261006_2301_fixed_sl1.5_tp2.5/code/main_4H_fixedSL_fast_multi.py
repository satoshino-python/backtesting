"""
複数通貨ペアのバックテストを、高速エンジン（fast_engine.py）と並列実行で行う。

main_4H_fixedSL_multi.py と同じ戦略を、backtesting.py の代わりに fast_engine.run_fast() で検証する。
結果は backtesting.py と一致する（compare_fast_engine.py で確認済み）。1ペア5年分が数秒で終わる。
  - EXIT_MODE="fixed"      : SwingBreakoutStrategy1Min（SL/TP は ATR 倍率、建値ストップは breakeven_trigger_r）
  - EXIT_MODE="swing_trail": SwingBreakoutTrail1Min（1時間足スイングのトレーリングストップ。TP・建値ストップなし）

- 通貨ペアごとに別プロセスで並列に実行する（WORKERS 個まで同時）
- 集計は main_4H_fixedSL_multi.py と同じ（ペア別 / 総合 / 年別を R倍数で）
- チャートはトレード付きのダウ理論チャート（dow_swing_chart.make_chart）をペアごとに出力する
  （backtesting.py 標準のチャートは backtesting.py の実行が必要で遅いため出力しない）
- 結果は RESULTS_ROOT/<日時>_<RUN_LABEL>/ に保存する（前回の結果を上書きしない）
"""
import io
import json
import shutil
import dataclasses
import contextlib
import time
import warnings
import webbrowser
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (
    Config,
    load_gmo_click_1min_data,
    resample_to_signal_bars,
    compute_signals,
    map_signals_to_1min,
    get_trading_day_label,
    add_extra_stats,
)
from main_4H_fixedSL_multi import (
    zip_month_range, price_decimals_for, r_stats, max_drawdown, print_table, compute_h1_swings,
)
from fast_engine import run_fast
from dow_swing_chart import make_chart


# ===== 変更するパラメータはここだけ =====
# 全ペア共通の設定（data_path / price_decimals はペアごとに自動で上書きされる）
BASE_CFG = Config(
    start_date="2021-01-01",
    end_date="2025-12-31",
    sl_atr_multiplier=1.5,
    tp_atr_multiplier=2.5,
    breakeven_trigger_r=None,     # None: 建値ストップなし / 1.0: 1R 到達で SL を建値へ
)
DATA_ROOT = Path("histData")      # この下の通貨ペア名フォルダ（histData/EURUSD など）を読み込む
PAIRS = None                      # None: DATA_ROOT 内のフォルダを全て検証 / 例: ("EURUSD", "USDJPY")
# 決済ルール（main_4H_fixedSL_multi.py と同じ）
# "fixed"      : SL = ATR × sl_atr_multiplier、TP = ATR × tp_atr_multiplier、建値ストップは breakeven_trigger_r
# "swing_trail": 1時間足スイングのトレーリングストップ（TP・建値ストップなし）。当初SLは INITIAL_SL_RULE
EXIT_MODE = "fixed"
H1_SWING_WINDOW = 5               # 1時間足スイングの前後判定本数（swing_trail のときだけ使う）
INITIAL_SL_RULE = "atr"           # "atr": ATR固定 / "near": ATRとスイングの近い方 / "far": 遠い方（swing_trail のみ）
# 結果は実行ごとに RESULTS_ROOT/<日時>_<RUN_LABEL>/ に保存する。
# RUN_LABEL には「何を変えた実行か」を短く書く（フォルダ名に使えない文字 \ / : * ? " < > | は不可）。
# None なら決済ルールと倍率から自動で付ける（例: fixed_sl1.5_tp2.5 / trailH1w5_atr_sl1.5）
RESULTS_ROOT = Path("fast_results")
RUN_LABEL = None
# 同時に実行するペア数。1ペアあたり最大 1〜2GB 程度のメモリを使うので、メモリに合わせて決める
WORKERS = 4
MAKE_CHARTS = True                # True: ペアごとにトレード付きのダウ理論チャートを出力（1ペア数十秒）
OPEN_CHARTS = False               # True: 出力したチャートをブラウザで開く（ペアの数だけタブが開く）


def discover_pairs():
    if PAIRS:
        return list(PAIRS)
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def run_label():
    if RUN_LABEL:
        return RUN_LABEL
    if EXIT_MODE == "swing_trail":
        return f"trailH1w{H1_SWING_WINDOW}_{INITIAL_SL_RULE}_sl{BASE_CFG.sl_atr_multiplier}"
    label = f"fixed_sl{BASE_CFG.sl_atr_multiplier}_tp{BASE_CFG.tp_atr_multiplier}"
    if BASE_CFG.breakeven_trigger_r is not None:
        label += f"_be{BASE_CFG.breakeven_trigger_r}"
    return label


def run_pair(pair, cfg, out_dir, make_charts, exit_mode, h1_window, initial_sl_rule):
    """
    1つの通貨ペアを検証する（別プロセスで実行される）。
    画面への出力は log_<ペア>.txt に保存し、集計に必要な小さなデータだけを返す。
    """
    log = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(log), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start_ts = pd.Timestamp(cfg.start_date) if cfg.start_date else None
        end_ts = pd.Timestamp(cfg.end_date) if cfg.end_date else None

        df_1min, avg_relative_spread = load_gmo_click_1min_data(
            cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
            month_range=zip_month_range(cfg),
            spread_period=(start_ts, end_ts),
            session_start_hour=cfg.session_start_hour,
        )
        signal_df = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
        signals = compute_signals(signal_df, window=cfg.window, atr_period=cfg.atr_period,
                                  price_decimals=cfg.price_decimals)
        df_ext = map_signals_to_1min(df_1min, signals, cfg.session_start_hour, cfg.signal_hours)
        if exit_mode == "swing_trail":
            df_ext = df_ext.join(compute_h1_swings(df_1min, h1_window, cfg.session_start_hour))
        day = get_trading_day_label(df_ext.index, cfg.session_start_hour)
        mask = np.ones(len(df_ext), dtype=bool)
        if start_ts is not None:
            mask &= day >= start_ts
        if end_ts is not None:
            mask &= day <= end_ts
        df = df_ext.loc[mask]
        del df_ext, signal_df
        if df.empty:
            raise ValueError("指定した検証期間にデータがありません")

        commission_rate = avg_relative_spread / 2

        def commission_func(order_size, price):
            return round(abs(order_size) * price * commission_rate, cfg.price_decimals)

        t_load = time.perf_counter() - t0
        params = dict(sl_atr_multiplier=cfg.sl_atr_multiplier, price_decimals=cfg.price_decimals,
                      risk_pct=cfg.risk_pct, exit_mode=exit_mode)
        if exit_mode == "swing_trail":
            params.update(initial_sl_rule=initial_sl_rule)
            mode_text = f"swing_trail (H1 w{h1_window}, 当初SL {initial_sl_rule} / ATR {cfg.sl_atr_multiplier})"
        else:
            params.update(tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=cfg.breakeven_trigger_r)
            mode_text = (f"fixed (SL {cfg.sl_atr_multiplier} / TP {cfg.tp_atr_multiplier} ATR"
                         f"{'' if cfg.breakeven_trigger_r is None else f' / 建値 {cfg.breakeven_trigger_r}R'})")
        stats = add_extra_stats(run_fast(df, cash=cfg.cash, commission=commission_func,
                                         margin=cfg.margin, **params))
        t_bt = time.perf_counter() - t0 - t_load
        print(stats)

        risk_amount = cfg.cash * cfg.risk_pct  # 1R（決済通貨建て）
        trades = stats["_trades"].copy()
        trades.insert(0, "Pair", pair)
        trades["R"] = trades["PnL"] / risk_amount
        r_curve = (stats["_equity_curve"]["Equity"] - cfg.cash) / risk_amount

        chart_path = None
        if make_charts:
            first_day, last_day = get_trading_day_label(df.index, cfg.session_start_hour)[[0, -1]]
            chart_path = out_dir / f"dow_trade_chart_{pair}_{first_day:%Y%m%d}-{last_day:%Y%m%d}_{exit_mode}.html"
            make_chart(
                df_1min, chart_path,
                f"{pair} {first_day:%Y-%m-%d} ~ {last_day:%Y-%m-%d} / トレード: {mode_text}",
                first_day, last_day,
                trades=stats["_trades"],
                weekly_trend=None,
                mode=exit_mode, risk=risk_amount,
                entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
                session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours,
            )
        t_total = time.perf_counter() - t0

    (out_dir / f"log_{pair}.txt").write_text(log.getvalue(), encoding="utf-8")
    public_stats = stats[[k for k in stats.index if not str(k).startswith("_")]]
    return dict(
        pair=pair,
        stats=public_stats,
        trades=trades,
        r_curve_hourly=r_curve.resample("1h").last().dropna(),
        max_dd_r=max_drawdown(r_curve),
        chart_path=chart_path,
        seconds=dict(load=t_load, backtest=t_bt, total=t_total),
    )


def save_run_info(out_dir, pairs):
    """この実行の設定（config.json）と実行時点のコード（code/）を保存する"""
    info = {
        "run_label": run_label(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "engine": "fast_engine",
        "exit_mode": EXIT_MODE,
        "h1_swing_window": H1_SWING_WINDOW if EXIT_MODE == "swing_trail" else None,
        "initial_sl_rule": INITIAL_SL_RULE if EXIT_MODE == "swing_trail" else None,
        "pairs": pairs,
        "base_cfg": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in dataclasses.asdict(BASE_CFG).items()},
    }
    (out_dir / "config.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code_dir = out_dir / "code"
    code_dir.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for name in (Path(__file__).name, "fast_engine.py", "main_4H_fixedSL.py"):
        shutil.copy2(here / name, code_dir / name)


def main():
    if EXIT_MODE not in ("fixed", "swing_trail"):
        print(f'❌ EXIT_MODE は "fixed" か "swing_trail" を指定してください: {EXIT_MODE}')
        return
    if BASE_CFG.execution_timeframe != "1min":
        print('❌ このスクリプトは execution_timeframe="1min" 専用です')
        return
    pairs = discover_pairs()
    if not pairs:
        print(f"❌ {DATA_ROOT} に通貨ペアのフォルダがありません。")
        return
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{run_label()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_info(out_dir, pairs)
    print(f"ℹ️ 検証する通貨ペア: {', '.join(pairs)}（同時に {min(WORKERS, len(pairs))} ペアずつ）")
    if EXIT_MODE == "swing_trail":
        rule = (f"決済 swing_trail（1時間足 前後{H1_SWING_WINDOW}本 / 当初SL {INITIAL_SL_RULE} / "
                f"ATR × {BASE_CFG.sl_atr_multiplier}）")
    else:
        rule = (f"決済 fixed（SL {BASE_CFG.sl_atr_multiplier} / TP {BASE_CFG.tp_atr_multiplier} / "
                f"建値ストップ {BASE_CFG.breakeven_trigger_r}）")
    print(f"ℹ️ 期間: {BASE_CFG.start_date} ～ {BASE_CFG.end_date} / {rule}")

    t0 = time.perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(pairs))) as ex:
        futures = {}
        for pair in pairs:
            cfg = dataclasses.replace(BASE_CFG, data_path=DATA_ROOT / pair,
                                      price_decimals=price_decimals_for(pair))
            futures[ex.submit(run_pair, pair, cfg, out_dir, MAKE_CHARTS,
                              EXIT_MODE, H1_SWING_WINDOW, INITIAL_SL_RULE)] = pair
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"❌ {pair}: エラーのためスキップします: {e!r}")
                continue
            results[pair] = res
            s = res["seconds"]
            print(f"✅ {pair}: {res['stats']['# Trades']} トレード / 合計 {res['trades']['R'].sum():+.1f}R"
                  f"（読み込み {s['load']:.1f}秒 / 検証 {s['backtest']:.1f}秒 / 全体 {s['total']:.1f}秒）")
    print(f"⏱️ 全ペアの処理時間: {time.perf_counter() - t0:.1f} 秒")

    if not results:
        print("❌ 検証できた通貨ペアがありません。")
        return
    results = {p: results[p] for p in pairs if p in results}  # 表示順をペア名順にそろえる

    # ===== 通貨ペア別の成績 =====
    rows = {}
    for pair, res in results.items():
        t = res["trades"].sort_values(["ExitTime", "EntryTime"], kind="stable")
        row = r_stats(t["R"])
        row["最大DD [R]"] = res["max_dd_r"]
        row["収益率 [%]"] = res["stats"]["Return [%]"]
        row["最大DD [%]"] = res["stats"]["Max. Drawdown [%]"]
        rows[pair] = row

    # ===== 総合成績（全トレードを決済順に並べて集計） =====
    all_trades = pd.concat([r["trades"] for r in results.values()], ignore_index=True)
    all_trades = all_trades.sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    total = r_stats(all_trades["R"])
    # 合算の資産曲線: 各ペアの累積R（含み損益込み・1時間ごと）を時刻でそろえて足し合わせる
    combined_curve = (
        pd.concat({p: r["r_curve_hourly"] for p, r in results.items()}, axis=1, sort=True)
        .sort_index().ffill().fillna(0).sum(axis=1)
    )
    total["最大DD [R]"] = max_drawdown(combined_curve)
    total["収益率 [%]"] = total["合計R"] * BASE_CFG.risk_pct / len(results) * 100
    total["最大DD [%]"] = float("nan")  # 口座ごとの%は合算できないため R で見る
    rows["合計"] = total

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary[["トレード数", "最大連敗"]] = summary[["トレード数", "最大連敗"]].astype(int)
    summary.index.name = "通貨ペア"
    print_table("通貨ペア別 / 総合成績（R倍数）", summary)

    # ===== 年別 × 通貨ペア別 =====
    all_trades["Year"] = pd.to_datetime(all_trades["ExitTime"]).dt.year
    yearly_r = all_trades.pivot_table(index="Year", columns="Pair", values="R",
                                      aggfunc="sum", fill_value=0.0)
    yearly_r["合計"] = yearly_r.sum(axis=1)
    yearly_n = all_trades.pivot_table(index="Year", columns="Pair", values="R",
                                      aggfunc="count", fill_value=0)
    yearly_n["合計"] = yearly_n.sum(axis=1)
    yearly_wr = all_trades.assign(Win=all_trades["R"] > 0).pivot_table(
        index="Year", columns="Pair", values="Win", aggfunc="mean") * 100
    yearly_wr["合計"] = all_trades.groupby("Year")["R"].apply(lambda s: (s > 0).mean() * 100)
    print_table("年別 合計R", yearly_r)
    print_table("年別 トレード数", yearly_n)
    print_table("年別 勝率 [%]", yearly_wr)

    # ===== CSV 出力 =====
    summary.round(4).to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    yearly = pd.concat({"合計R": yearly_r, "トレード数": yearly_n, "勝率[%]": yearly_wr}, axis=1)
    yearly.round(4).to_csv(out_dir / "yearly.csv", encoding="utf-8-sig")
    out_trades = all_trades.drop(columns="Year")
    out_trades["R"] = out_trades["R"].round(4)
    out_trades["ReturnPct"] = out_trades["ReturnPct"].round(8)
    out_trades.to_csv(out_dir / "trades_all.csv", index=False, encoding="utf-8-sig")
    for pair, res in results.items():
        res["trades"].to_csv(out_dir / f"trades_{pair}.csv", index=False, encoding="utf-8-sig")
    combined_curve.rename("CumR").to_csv(out_dir / "equity_R_combined.csv", encoding="utf-8-sig")
    pd.DataFrame({p: r["stats"] for p, r in results.items()}).to_csv(
        out_dir / "stats_by_pair.csv", encoding="utf-8-sig")

    print(f"\n💾 出力先: {out_dir.resolve()}")
    print("   summary.csv（ペア別・総合成績） / yearly.csv（年別） / trades_all.csv・trades_<ペア>.csv（取引履歴・R列付き）")
    print("   stats_by_pair.csv（backtesting.py 形式の統計） / equity_R_combined.csv（合算の累積R・1時間ごと）")
    if MAKE_CHARTS:
        print("   dow_trade_chart_<ペア>_*.html（トレード付きダウ理論チャート） / log_<ペア>.txt（ペアごとのログ）")
        if OPEN_CHARTS:
            for res in results.values():
                if res["chart_path"]:
                    webbrowser.open(res["chart_path"].resolve().as_uri(), new=2)


if __name__ == "__main__":
    main()
