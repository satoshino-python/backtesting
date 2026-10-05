"""
複数通貨ペアのバックテスト（main_4H_fixedSL.py の戦略・データ読み込みをそのまま使う）

通貨ペアごとに独立した口座（同じ初期資金・同じ risk_pct）として順番にバックテストし、
  - 通貨ペア別の成績
  - 全ペア合算の総合成績
  - 年別 × 通貨ペア別の成績
を R倍数で集計する。チャート（HTML）は通貨ペアごとに出力する。

【R倍数での合算について】
backtesting.py の損益は「決済通貨」建て（EURUSD なら USD、USDJPY なら JPY）のため、
金額のままでは通貨ペア間で足し合わせられない。
この戦略は 1回のトレードの損失額を「初期資金 × risk_pct」（＝1R）に固定しているので、
    R倍数 = 損益 ÷ (初期資金 × risk_pct)
とすれば通貨に依存しない共通の単位になる（損切りで約 -1R、利確で約 +TP倍率/SL倍率 R）。
合計R × risk_pct が「初期資金に対する損益率」の目安になる（例: +50R × 2% = +100%）。

【注意】各ペアは別口座扱いのため、複数ペアを同時保有したときの証拠金の取り合いは再現しない。
"""
import os
import gc
import json
import shutil
import dataclasses
from datetime import datetime
import warnings
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

from main_4H_fixedSL import (
    Config,
    load_gmo_click_1min_data,
    resample_to_signal_bars,
    signal_bar_trading_day,
    get_trading_day_label,
    compute_signals,
    map_signals_to_1min,
    make_chart_index_for_signal_bars,
    SwingBreakoutStrategy,
    SwingBreakoutStrategy1Min,
    add_extra_stats,
    append_stats_to_html,
)


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
# 結果は実行ごとに RESULTS_ROOT/<日時>_<RUN_LABEL>/ に保存する（前回の結果を上書きしない）。
# RUN_LABEL には「何を変えた実行か」を短く書く（フォルダ名に使えない文字 \ / : * ? " < > | は不可）
RESULTS_ROOT = Path("multi_results")
RUN_LABEL = "noBE"
OUTPUT_DIR = None                 # main() で決まる（テスト等で固定したい場合だけ Path を指定）
OPEN_CHARTS = True                # True: ペアごとのチャートをブラウザで開く（ペアの数だけタブが開く）
# 検証期間の開始月より何ヶ月前の ZIP から読み込むか（SwingHigh/Low・ATR の助走期間）。
# 4時間足・window=18 なら Swing の確定に約3日、ATR(EMA 18本)が落ち着くのに約10日かかるため、
# 1ヶ月あれば十分。window や atr_period を大きくした場合は増やすこと。
WARMUP_MONTHS = 1


def price_decimals_for(pair):
    """クロス円（JPY建て）は小数3桁、それ以外は5桁（GMOクリック証券の表示桁数）"""
    return 3 if pair.upper().endswith("JPY") else 5


def discover_pairs():
    if PAIRS:
        return list(PAIRS)
    return sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())


def zip_month_range(cfg):
    """
    検証期間から、読み込む ZIP の年月範囲（YYYYMM の整数）を決める。
    - 開始: 検証開始月の WARMUP_MONTHS ヶ月前（助走期間）
    - 終了: 検証終了月の翌月。GMOのデータは日本時間で記録されており、NY 17:00 区切りの
            取引日の最終バーは日本時間の翌朝になるため、月末が終了日だと翌月のZIPに入る。
    start_date / end_date が None の側は制限しない。
    """
    lo = hi = None
    if cfg.start_date:
        p = pd.Timestamp(cfg.start_date).to_period("M") - WARMUP_MONTHS
        lo = p.year * 100 + p.month
    if cfg.end_date:
        p = pd.Timestamp(cfg.end_date).to_period("M") + 1
        hi = p.year * 100 + p.month
    return lo, hi


def max_drawdown(curve):
    """累積R（または累積損益）の曲線から最大ドローダウン（ピークからの最大下落幅・正の値）を返す"""
    if len(curve) == 0:
        return 0.0
    return float((curve.cummax() - curve).max())


def r_stats(r):
    """R倍数の配列（決済順）から成績指標を計算する"""
    r = np.asarray(r, dtype=float)
    nan = float("nan")
    wins, losses = r[r > 0], r[r < 0]
    streak = max_streak = 0
    for v in r:
        streak = streak + 1 if v < 0 else 0
        max_streak = max(max_streak, streak)
    return {
        "トレード数": len(r),
        "勝率 [%]": (len(wins) / len(r) * 100) if len(r) else nan,
        "合計R": float(r.sum()),
        "平均R": float(r.mean()) if len(r) else nan,
        "平均勝ちR": float(wins.mean()) if len(wins) else nan,
        "平均負けR": float(losses.mean()) if len(losses) else nan,
        "プロフィットファクター": float(wins.sum() / abs(losses.sum())) if len(losses) else nan,
        "ペイオフレシオ": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else nan,
        "最大連敗": int(max_streak),
    }


def run_pair(pair, cfg):
    """
    1つの通貨ペアをバックテストし、チャートを出力する。
    main_4H_fixedSL.main() の単発検証と同じ処理（グリッド検証は非対応）。

    Returns: dict(stats, trades, r_curve, r_curve_hourly)
      r_curve        : 資産曲線を R 単位に換算したもの（含み損益込み・元の解像度）
      r_curve_hourly : 合算用に1時間ごとの最終値へ間引いたもの
    """
    print(f"\n{'#' * 70}\n# {pair}（price_decimals={cfg.price_decimals}）\n{'#' * 70}")

    start_ts = pd.Timestamp(cfg.start_date) if cfg.start_date else None
    end_ts = pd.Timestamp(cfg.end_date) if cfg.end_date else None

    # 検証期間（＋助走期間）の ZIP だけを読み込み、平均スプレッドは検証期間だけで計算する
    df_1min, avg_relative_spread = load_gmo_click_1min_data(
        cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
        month_range=zip_month_range(cfg),
        spread_period=(start_ts, end_ts),
        session_start_hour=cfg.session_start_hour,
    )
    signal_df = resample_to_signal_bars(
        df_1min, session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours,
    )

    commission_rate = avg_relative_spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * commission_rate, cfg.price_decimals)

    def in_period(day_labels):
        mask = np.ones(len(day_labels), dtype=bool)
        if start_ts is not None:
            mask &= day_labels >= start_ts
        if end_ts is not None:
            mask &= day_labels <= end_ts
        return mask

    if cfg.execution_timeframe == "signal":
        df = signal_df.loc[in_period(signal_bar_trading_day(signal_df.index, cfg.session_start_hour))]
        strategy = SwingBreakoutStrategy
        run_params = dict(window=cfg.window, atr_period=cfg.atr_period,
                          swing_point_style=cfg.swing_point_style)
    else:
        signals = compute_signals(signal_df, window=cfg.window, atr_period=cfg.atr_period,
                                  price_decimals=cfg.price_decimals)
        df_ext = map_signals_to_1min(df_1min, signals, session_start_hour=cfg.session_start_hour,
                                     signal_hours=cfg.signal_hours)
        df = df_ext.loc[in_period(get_trading_day_label(df_ext.index, cfg.session_start_hour))]
        strategy = SwingBreakoutStrategy1Min
        run_params = {}
        del df_ext
    del df_1min, signal_df
    gc.collect()

    if df.empty:
        print(f"⚠️ {pair}: 指定した検証期間にデータがありません。スキップします。")
        return None

    run_params.update(
        sl_atr_multiplier=cfg.sl_atr_multiplier,
        tp_atr_multiplier=cfg.tp_atr_multiplier,
        breakeven_trigger_r=cfg.breakeven_trigger_r,
        price_decimals=cfg.price_decimals,
        risk_pct=cfg.risk_pct,
    )

    print(f"\n================ {pair} バックテスト実行中 ================")
    bt = Backtest(df, strategy, cash=cfg.cash, commission=commission_func,
                  margin=cfg.margin, exclusive_orders=False)
    stats = add_extra_stats(bt.run(**run_params))
    print(stats)

    risk_amount = cfg.cash * cfg.risk_pct  # 1R（決済通貨建て）
    trades = stats["_trades"].copy()
    trades.insert(0, "Pair", pair)
    trades["R"] = trades["PnL"] / risk_amount

    r_curve = (stats["_equity_curve"]["Equity"] - cfg.cash) / risk_amount
    r_curve_hourly = r_curve.resample("1h").last().dropna()

    # ===== チャート出力（main_4H_fixedSL.main() と同じ表示ロジック） =====
    html_path = OUTPUT_DIR / f"chart_{pair}.html"
    plot_bt = bt
    if cfg.execution_timeframe == "signal":
        chart_resample = False
    elif cfg.chart_timeframe == "signal":
        chart_df = df.copy()
        chart_df.index = make_chart_index_for_signal_bars(df.index, cfg.session_start_hour, cfg.signal_hours)
        plot_bt = Backtest(chart_df, strategy, cash=cfg.cash, commission=commission_func,
                           margin=cfg.margin, exclusive_orders=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 同じ警告が上の実行で出ているため繰り返さない
            plot_stats = plot_bt.run(**run_params)
        if plot_stats["# Trades"] != stats["# Trades"]:
            print("⚠️ チャート用の再実行結果とトレード数が一致しません。チャートの表示内容にご注意ください。")
        chart_resample = f"{cfg.signal_hours}h"
    else:
        chart_resample = False if len(df) <= 20_000 else True

    print(f"\n📊 チャートを生成中... ({html_path})")
    plot_bt.plot(filename=str(html_path), open_browser=False,
                 resample=chart_resample, plot_volume=False)
    append_stats_to_html(html_path, stats, title=f"Stats ({pair})")
    if OPEN_CHARTS:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path), new=2)
        except Exception:
            pass

    return dict(stats=stats, trades=trades, r_curve=r_curve, r_curve_hourly=r_curve_hourly)


def save_run_info(pairs):
    """
    結果フォルダに、この実行の設定（config.json）と実行時点のコード（code/）を保存する。
    後から「この結果はどの設定・どのコードで出したものか」を確実に辿れるようにするため。
    """
    info = {
        "run_label": RUN_LABEL,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "pairs": pairs,
        "warmup_months": WARMUP_MONTHS,
        "base_cfg": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in dataclasses.asdict(BASE_CFG).items()},
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    code_dir = OUTPUT_DIR / "code"
    code_dir.mkdir(exist_ok=True)
    here = Path(__file__).resolve().parent
    for name in (Path(__file__).name, "main_4H_fixedSL.py"):
        shutil.copy2(here / name, code_dir / name)


def print_table(title, table):
    print(f"\n================ {title} ================")
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 250, "display.max_columns", None):
        print(table.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="-"))


def main():
    if BASE_CFG.signal_hours <= 0 or 24 % BASE_CFG.signal_hours != 0:
        print(f"❌ signal_hours は 24 の約数を指定してください: {BASE_CFG.signal_hours}")
        return
    if BASE_CFG.grid_enabled:
        print("ℹ️ 複数ペア検証では grid_enabled は無視されます（単発検証のみ）。")

    pairs = discover_pairs()
    if not pairs:
        print(f"❌ {DATA_ROOT} に通貨ペアのフォルダがありません。")
        return
    print(f"ℹ️ 検証する通貨ペア: {', '.join(pairs)}")
    global OUTPUT_DIR
    if OUTPUT_DIR is None:
        OUTPUT_DIR = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_run_info(pairs)

    results = {}
    for pair in pairs:
        cfg = dataclasses.replace(BASE_CFG, data_path=DATA_ROOT / pair,
                                  price_decimals=price_decimals_for(pair))
        try:
            res = run_pair(pair, cfg)
        except Exception as e:
            print(f"❌ {pair}: エラーのためスキップします: {e}")
            continue
        if res is not None:
            results[pair] = res
        gc.collect()

    if not results:
        print("❌ 検証できた通貨ペアがありません。")
        return

    # ===== 通貨ペア別の成績 =====
    rows = {}
    for pair, res in results.items():
        t = res["trades"].sort_values(["ExitTime", "EntryTime"], kind="stable")
        row = r_stats(t["R"])
        row["最大DD [R]"] = max_drawdown(res["r_curve"])
        row["収益率 [%]"] = res["stats"]["Return [%]"]
        row["最大DD [%]"] = res["stats"]["Max. Drawdown [%]"]
        rows[pair] = row

    # ===== 総合成績（全トレードを決済順に並べて集計） =====
    all_trades = pd.concat([r["trades"] for r in results.values()], ignore_index=True)
    all_trades = all_trades.sort_values(["ExitTime", "EntryTime"], kind="stable").reset_index(drop=True)
    total = r_stats(all_trades["R"])

    # 合算の資産曲線: 各ペアの累積R（含み損益込み・1時間ごと）を時刻でそろえて足し合わせる
    combined_curve = (
        pd.concat({p: r["r_curve_hourly"] for p, r in results.items()}, axis=1)
        .sort_index().ffill().fillna(0).sum(axis=1)
    )
    total["最大DD [R]"] = max_drawdown(combined_curve)
    # 全ペア合計の初期資金に対する収益率（各ペアの初期資金は同額）
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
    summary.round(4).to_csv(OUTPUT_DIR / "summary.csv", encoding="utf-8-sig")
    yearly = pd.concat({"合計R": yearly_r, "トレード数": yearly_n, "勝率[%]": yearly_wr}, axis=1)
    yearly.round(4).to_csv(OUTPUT_DIR / "yearly.csv", encoding="utf-8-sig")
    out_trades = all_trades.drop(columns="Year")
    out_trades["R"] = out_trades["R"].round(4)
    out_trades["ReturnPct"] = out_trades["ReturnPct"].round(8)
    out_trades.to_csv(OUTPUT_DIR / "trades_all.csv", index=False, encoding="utf-8-sig")
    combined_curve.rename("CumR").to_csv(OUTPUT_DIR / "equity_R_combined.csv", encoding="utf-8-sig")

    print(f"\n💾 出力先: {OUTPUT_DIR.resolve()}")
    print("   summary.csv（ペア別・総合成績） / yearly.csv（年別） / trades_all.csv（全取引・R列付き）")
    print("   equity_R_combined.csv（合算の累積R・1時間ごと） / chart_<ペア>.html（ペアごとのチャート）")


if __name__ == "__main__":
    main()
