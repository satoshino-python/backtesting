"""
main_4H_fixedSL.py の戦略に、週足のダウ理論トレンドフィルター（dow_trend.py）を加えた版。

エントリー・決済・枚数の決め方は main_4H_fixedSL.py と同じ（関数・Strategy をそのまま import して使う）。
違いは「週足のトレンド判定と同じ方向の逆指値だけを置く」ことだけ。

【週足の作り方】
NY 17:00 区切りの取引日（月〜金）を、金曜日で終わる1週間にまとめる（日曜 17:00 ～ 金曜 17:00）。
トレンド判定は docs/DOW_TREND_SPEC.md の compute_dow_trend（1=上昇, 0=レンジ, -1=下降）。

【ルックアヘッド防止】
週足の判定はその週が終わるまで確定しないため、各週のトレードには「前の週が終わった時点の判定」を使う
（4時間足シグナルを1本シフトしているのと同じ考え方）。

【比較】
同じデータで「フィルターなし」と各フィルターのモードを続けて実行し、成績を1つの表にまとめる。
"""
import os
import warnings
import webbrowser
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

from dow_swing_chart import make_chart
from dow_trend import DowConfig, compute_dow_trend
from main_4H_fixedSL import (
    Config,
    load_gmo_click_1min_data,
    resample_to_signal_bars,
    get_trading_day_label,
    compute_signals,
    map_signals_to_1min,
    make_chart_index_for_signal_bars,
    SwingBreakoutStrategy1Min,
    add_extra_stats,
    append_stats_to_html,
)


# ===== 変更するパラメータはここだけ =====
@dataclass(frozen=True)
class DowFilterConfig(Config):
    # 週足ダウ理論の判定パラメータ（dow_trend.DowConfig と同じ意味。単位は週足の本数）
    dow_n: int = 3                   # ピボット幅（左右 n 週）
    dow_atr_period: int = 14         # ATR 期間（週）
    dow_min_swing_atr: float = 1.0   # 最小スイング幅（週足ATRの倍数）
    dow_use_wick: bool = True        # True: ヒゲでブレイク判定 / False: 終値で判定
    # フィルターの種類
    # "strict"    : 上昇なら買いだけ、下降なら売りだけ。レンジ（判定なし）のときは取引しない
    # "no_counter": トレンドに逆らう方向だけ止める（上昇中は売りなし、下降中は買いなし。レンジは両方向）
    dow_mode: str = "strict"
    # 比較表に並べるモード（None はフィルターなし＝main_4H_fixedSL.py と同じ結果）
    compare_modes: tuple = (None, "strict", "no_counter")
    result_dir: str = "dow_results"  # 取引履歴CSV・比較表・チャートの保存先


CFG = DowFilterConfig(
    start_date="2025-01-01",
    end_date="2025-12-31",
    sl_atr_multiplier=1.5,
    tp_atr_multiplier=2.5,
)
OPEN_CHART = True  # True: dow_mode のチャートをブラウザで開く
# True: dow_mode のトレードを載せたダウ理論チャート（dow_swing_chart.py）も出力する
# （dow_trade_chart_<通貨>_<期間>_<モード>.html。エントリー/決済・SL/TP・使った週足トレンド・エントリーライン）
TRADE_CHART = True


def resample_to_weekly_bars(df_1min, session_start_hour=17):
    """
    1分足（NY時間・TZ付き）を、NY 17:00 区切りの取引日ベースの週足にまとめる。
    インデックスはその週の最終取引日（金曜日）の日付（タイムゾーンなし）。
    """
    week = week_label(df_1min.index, session_start_hour)
    bars = df_1min.groupby(week).agg(
        Open=("Open", "first"), High=("High", "max"),
        Low=("Low", "min"), Close=("Close", "last"),
    )
    bars.index.name = "WeekEnd"
    return bars


def week_label(index_ny, session_start_hour=17):
    """各バーが属する週（金曜日で終わる週）の最終日の日付"""
    trading_day = get_trading_day_label(index_ny, session_start_hour)
    return trading_day.to_period("W-FRI").end_time.normalize()


def map_weekly_trend_to_1min(df_1min_ext, weekly_trend, session_start_hour=17):
    """
    週足トレンドを1本（1週）シフトし、1分足の各バーに DowTrend 列として付ける。
    これで各週の1分足は「前の週の終値時点で確定していた判定」を使う。
    """
    shifted = weekly_trend.shift(1)
    week = week_label(df_1min_ext.index, session_start_hour)
    out = df_1min_ext.copy()
    out["DowTrend"] = shifted.reindex(week).fillna(0).to_numpy()
    return out


class SwingBreakoutDowFilter1Min(SwingBreakoutStrategy1Min):
    """
    SwingBreakoutStrategy1Min と同じ注文を置いたあと、週足トレンドに合わない方向の
    未約定の逆指値を取り消す。dow_mode=None ならフィルターなし（元の戦略と同じ）。
    """
    dow_mode = None

    def init(self):
        super().init()
        self.dow_trend = self.I(lambda: self.data.DowTrend, overlay=False, name="Dow Trend (Weekly)")

    def next(self):
        super().next()
        if self.dow_mode is None or self.position:
            return
        trend = self.dow_trend[-1]
        for order in self.orders:
            if order.is_contingent:
                continue
            if self.dow_mode == "strict":
                allowed = trend == 1 if order.is_long else trend == -1
            else:  # "no_counter"
                allowed = trend != -1 if order.is_long else trend != 1
            if not allowed:
                order.cancel()


def summarize(stats, cfg):
    """比較表の1行分（R倍数は 1R = 初期資金 × risk_pct で換算）"""
    trades = stats["_trades"]
    r = trades["PnL"] / (cfg.cash * cfg.risk_pct)
    wins, losses = r[r > 0], r[r < 0]
    return {
        "トレード数": len(r),
        "買い": int((trades["Size"] > 0).sum()),
        "売り": int((trades["Size"] < 0).sum()),
        "勝率 [%]": (r > 0).mean() * 100 if len(r) else np.nan,
        "合計R": r.sum(),
        "平均R": r.mean() if len(r) else np.nan,
        "プロフィットファクター": wins.sum() / -losses.sum() if len(losses) else np.nan,
        "ペイオフレシオ": stats["Payoff Ratio"],
        "最大連敗": stats["Max. Consecutive Losses"],
        "収益率 [%]": stats["Return [%]"],
        "最大DD [%]": stats["Max. Drawdown [%]"],
        "シャープレシオ": stats["Sharpe Ratio"],
    }


def main():
    if CFG.execution_timeframe != "1min":
        print('❌ この版は execution_timeframe="1min" のみ対応しています。')
        return
    out_dir = Path(CFG.result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_1min, avg_relative_spread = load_gmo_click_1min_data(
        CFG.data_path, price_side=CFG.price_side, price_decimals=CFG.price_decimals,
    )
    signal_df = resample_to_signal_bars(
        df_1min, session_start_hour=CFG.session_start_hour, signal_hours=CFG.signal_hours,
    )
    commission_rate = avg_relative_spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * commission_rate, CFG.price_decimals)

    # 週足トレンド（全期間で計算してから、検証期間に絞る）
    weekly = resample_to_weekly_bars(df_1min, CFG.session_start_hour)
    dow_cfg = DowConfig(n=CFG.dow_n, atr_period=CFG.dow_atr_period,
                        min_swing_atr=CFG.dow_min_swing_atr, use_wick=CFG.dow_use_wick)
    weekly_trend, pivots = compute_dow_trend(weekly, dow_cfg)
    print(f"✅ 週足 {len(weekly)} 本でダウ理論のトレンドを判定しました（採用スイング {len(pivots)} 個）")

    signals = compute_signals(signal_df, window=CFG.window, atr_period=CFG.atr_period,
                              price_decimals=CFG.price_decimals)
    df_ext = map_signals_to_1min(df_1min, signals, session_start_hour=CFG.session_start_hour,
                                 signal_hours=CFG.signal_hours)
    df_ext = map_weekly_trend_to_1min(df_ext, weekly_trend, CFG.session_start_hour)

    trading_day = get_trading_day_label(df_ext.index, CFG.session_start_hour)
    start_ts = pd.Timestamp(CFG.start_date) if CFG.start_date else trading_day.min()
    end_ts = pd.Timestamp(CFG.end_date) if CFG.end_date else trading_day.max()
    df = df_ext.loc[(trading_day >= start_ts) & (trading_day <= end_ts)]
    del df_ext
    if df.empty:
        print("❌ 指定した検証期間にデータがありません。")
        return
    period = f"{start_ts:%Y%m%d}-{end_ts:%Y%m%d}"

    # 検証期間の各週で使われる判定（＝前週末の判定）
    used = weekly_trend.shift(1).fillna(0).astype(int)
    used = used[(used.index >= start_ts) & (used.index <= end_ts + pd.Timedelta(days=6))]
    labels = {1: "上昇", 0: "レンジ", -1: "下降"}
    print("\n================ 検証期間の週足トレンド（各週に使った判定） ================")
    print(used.map(labels).value_counts().to_string())
    changes = used[used != used.shift(1)]
    for wk, v in changes.items():
        print(f"  {wk - pd.Timedelta(days=4):%Y-%m-%d} の週から {labels[v]}")
    used.map(labels).rename("trend").to_csv(out_dir / f"weekly_trend_{period}.csv", encoding="utf-8-sig")

    base_params = dict(
        sl_atr_multiplier=CFG.sl_atr_multiplier,
        tp_atr_multiplier=CFG.tp_atr_multiplier,
        breakeven_trigger_r=CFG.breakeven_trigger_r,
        price_decimals=CFG.price_decimals,
        risk_pct=CFG.risk_pct,
    )
    rows = {}
    chart_bt = None
    for mode in CFG.compare_modes:
        name = mode or "filter_off"
        print(f"\n================ バックテスト実行中（週足ダウフィルター: {name}） ================")
        bt = Backtest(df, SwingBreakoutDowFilter1Min, cash=CFG.cash, commission=commission_func,
                      margin=CFG.margin, exclusive_orders=False)
        stats = add_extra_stats(bt.run(dow_mode=mode, **base_params))
        rows[name] = summarize(stats, CFG)
        stats["_trades"].to_csv(out_dir / f"trades_{period}_{name}.csv", index=False,
                                float_format=f"%.{CFG.price_decimals}f")
        if mode == CFG.dow_mode:
            chart_bt, chart_stats = bt, stats

    table = pd.DataFrame(rows).T
    print("\n================ 比較（週足ダウ理論フィルター） ================")
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 250, "display.max_columns", None):
        print(table.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="-"))
    table.to_csv(out_dir / f"comparison_{period}.csv", encoding="utf-8-sig")
    print(f"\n💾 比較表・取引履歴・週足トレンドを {out_dir}/ に保存しました")

    if chart_bt is None:
        return
    if TRADE_CHART:
        symbol = Path(CFG.data_path).name
        trade_html = make_chart(
            df_1min, out_dir / f"dow_trade_chart_{symbol}_{period}_{CFG.dow_mode}.html",
            f"{symbol} {start_ts:%Y-%m-%d} ~ {end_ts:%Y-%m-%d} / トレード: {CFG.dow_mode}",
            start_ts, end_ts, trades=chart_stats["_trades"],
            weekly_trend=weekly_trend.shift(1).fillna(0).astype(int),  # 各週に使った判定（前週末の判定）
            mode=CFG.dow_mode, risk=CFG.cash * CFG.risk_pct,
            entry_window=CFG.window, entry_atr=CFG.atr_period, price_decimals=CFG.price_decimals,
            session_start_hour=CFG.session_start_hour, signal_hours=CFG.signal_hours,
        )
        if OPEN_CHART:
            try:
                webbrowser.open("file://" + os.path.abspath(trade_html), new=2)
            except Exception:
                pass
    # チャート（dow_mode の結果）。表示は main_4H_fixedSL.py と同じく判定足にまとめる
    html_path = out_dir / f"chart_{period}_{CFG.dow_mode}.html"
    chart_df = df.copy()
    chart_df.index = make_chart_index_for_signal_bars(df.index, CFG.session_start_hour, CFG.signal_hours)
    plot_bt = Backtest(chart_df, SwingBreakoutDowFilter1Min, cash=CFG.cash, commission=commission_func,
                       margin=CFG.margin, exclusive_orders=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_stats = plot_bt.run(dow_mode=CFG.dow_mode, **base_params)
    if plot_stats["# Trades"] != chart_stats["# Trades"]:
        print("⚠️ チャート用の再実行結果とトレード数が一致しません。チャートの表示内容にご注意ください。")
    plot_bt.plot(filename=str(html_path), open_browser=False,
                 resample=f"{CFG.signal_hours}h", plot_volume=False)
    append_stats_to_html(html_path, chart_stats, title=f"Stats (weekly Dow filter: {CFG.dow_mode})")
    print(f"📊 チャートを保存しました: {html_path}")
    if OPEN_CHART:
        try:
            webbrowser.open("file://" + os.path.abspath(html_path), new=2)
        except Exception:
            pass


if __name__ == "__main__":
    main()
