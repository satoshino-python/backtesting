"""
水平ライン逆張り（レンジ戦略）。4時間足スイングの高値・安値をレンジの上限・下限とみなし、そこで逆張りする。

【ライン】main_4H_fixedSL.py と同じ4時間足スイング（前後 window 本、既定18）
  - 上限 = 直近で確定した Swing High、下限 = 直近で確定した Swing Low
  - どちらも「確定してから一度も抜かれていない（有効）」ときだけ使う。抜かれたら次のスイングが確定するまで使わない
  - レンジ幅（上限 − 下限）が min_range_atr × ATR 未満なら取引しない（狭すぎるレンジは利幅が取れない）

【エントリー】entry_mode で選ぶ
  "limit"  : ラインに指値（ポジションが無いとき、毎バー置き直す）
    - 終値 < 上限 → 上限に売り指値 / 終値 > 下限 → 下限に買い指値（両方同時に置く）
    - 片方が約定したら、反対側の指値は取り消す（ブレイクアウト版のように反対側の注文で決済されることはない）
  "confirm": 反転を確認してから成行
    - 4時間足の高値が上限に届いた（抜けた）うえで、終値が上限より下で確定 → 次の4時間足の最初の1分足で売り
      （下限も同様に買い）。その4時間足が始まる時点で両方のラインが有効で、レンジ幅の条件を満たしていること
    - 発注は次の4時間足の最初の1分足の終値時点なので、約定はその次の1分足の始値（4時間足の始値から1分遅れ）
    - その時点でポジションを持っていれば見送る

【決済】
  - 利確 = エントリー価格から、レンジ幅 × tp_range_frac だけ内側（0.5 = レンジ中央、1.0 = 反対側のライン）
    tp_atr を指定すると、代わりに ATR × tp_atr だけ内側
  - 損切り = ラインの外側 ATR × sl_buffer_atr（"confirm" ではエントリー価格から ATR × sl_buffer_atr）
  - 建値ストップ（breakeven_trigger_r）は既定で無効

【枚数】損切り幅で損失が 初期資金 × risk_pct（1R）になるように決める（main_4H_fixedSL.py と同じ。複利なし）

【レンジ判定フィルター】ダウ理論（dow_trend.py）の判定が「レンジ（0）」のときだけエントリーする
  - "weekly": 週足の判定（前週末に確定したもの）。main_4H_fixedSL_dow.py と同じ週足
  - "h4"    : 4時間足の判定（1本前の4時間足の終値時点で確定したもの）
  - None    : フィルターなし
  1回の実行で compare_modes のモードを順番に検証し、比較表を作る。

結果は result_root/<日時>_<run_label>/ に保存する（前の結果を上書きしない）。
"""
import json
import os
import warnings
import webbrowser
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from dow_swing_chart import make_chart
from dow_trend import DowConfig, compute_dow_trend
from main_4H_fixedSL import (
    Config,
    load_gmo_click_1min_data,
    resample_to_signal_bars,
    get_trading_day_label,
    get_signal_bar_label,
    compute_signals,
    map_signals_to_1min,
    make_chart_index_for_signal_bars,
    move_sl_to_breakeven,
    add_extra_stats,
    append_stats_to_html,
)
from main_4H_fixedSL_dow import resample_to_weekly_bars, map_weekly_trend_to_1min, summarize


# ===== 変更するパラメータはここだけ =====
# Config の sl_atr_multiplier / tp_atr_multiplier はこの戦略では使わない
@dataclass(frozen=True)
class RangeReversalConfig(Config):
    entry_mode: str = "limit"        # "limit": ラインに指値 / "confirm": 4時間足で反転を確認してから成行
    sl_buffer_atr: float = 0.5       # 損切り = ラインの外側 ATR × この値（"confirm" ではエントリー価格から）
    tp_range_frac: float = 0.5       # 利確 = エントリーからレンジ幅 × この値だけ内側（0.5=中央、1.0=反対側のライン）
    tp_atr: float | None = None      # 指定すると利確 = エントリーから ATR × この値だけ内側（tp_range_frac より優先）
    min_range_atr: float = 2.0       # レンジ幅がこの値 × ATR 未満なら取引しない（0 で制限なし）
    breakeven_trigger_r: float | None = None  # 建値ストップ（R）。None で無効
    # 週足ダウ理論（main_4H_fixedSL_dow.py と同じ。単位は週足の本数）
    dow_n: int = 3
    dow_atr_period: int = 14
    dow_min_swing_atr: float = 1.0
    dow_use_wick: bool = True
    # 4時間足ダウ理論（単位は4時間足の本数）
    h4_dow_n: int = 3
    h4_dow_atr_period: int = 14
    h4_dow_min_swing_atr: float = 1.0
    h4_dow_use_wick: bool = True
    # レンジ判定フィルター: None（なし）/ "weekly"（週足がレンジのときだけ）/ "h4"（4時間足がレンジのときだけ）
    filter_mode: str | None = "weekly"  # チャートを作るモード
    compare_modes: tuple = (None, "weekly", "h4")  # 比較表に並べるモード
    result_root: str = "range_results"
    run_label: str = "rangeRev"


CFG = RangeReversalConfig(
    start_date="2025-01-01",
    end_date="2025-12-31",
)
OPEN_CHART = True  # True: チャートをブラウザで開く

FILTER_COLUMN = {"weekly": "DowTrend", "h4": "H4DowTrend"}


def map_h4_trend_to_1min(df_1min_ext, h4_trend, session_start_hour=17, signal_hours=4):
    """
    4時間足のダウ理論トレンドを1本シフトし、1分足の各バーに H4DowTrend 列として付ける
    （各4時間足の中では、1本前の4時間足の終値時点で確定していた判定を使う）。
    """
    label = get_signal_bar_label(df_1min_ext.index, session_start_hour, signal_hours)
    out = df_1min_ext.copy()
    out["H4DowTrend"] = h4_trend.shift(1).reindex(label).fillna(0).to_numpy()
    return out


def map_confirm_signals_to_1min(df_1min_ext, signal_df, signals, min_range_atr,
                                session_start_hour=17, signal_hours=4):
    """
    entry_mode="confirm" 用。4時間足 t でラインに届いて内側で確定したら、次の足 t+1 でエントリーするシグナルを作る。
    signals は compute_signals() の出力（1本シフト済み＝足 t の行は足 t-1 の終値時点のライン）なので、
    足 t の高値・安値・終値と比べてもルックアヘッドにならない。結果をさらに1本シフトして足 t+1 に割り当てる。

    1分足に ConfirmLong / ConfirmShort（0/1）と NewSignalBar（各4時間足の最初の1分足なら1）を付けて返す。
    """
    width = signals["SH"] - signals["SL"]
    intact = ((signals["SHValid"] == 1) & (signals["SLValid"] == 1) & (width > 0)
              & (width >= min_range_atr * signals["ATR"]))
    short = intact & (signal_df["High"] >= signals["SH"]) & (signal_df["Close"] < signals["SH"])
    long_ = intact & (signal_df["Low"] <= signals["SL"]) & (signal_df["Close"] > signals["SL"])
    both = short & long_  # 1本で上限・下限の両方に触れた足は見送る
    confirm = pd.DataFrame({"ConfirmLong": long_ & ~both, "ConfirmShort": short & ~both}).astype(float)
    confirm = confirm.shift(1).fillna(0)

    label = get_signal_bar_label(df_1min_ext.index, session_start_hour, signal_hours)
    mapped = confirm.reindex(label)
    out = df_1min_ext.copy()
    out["ConfirmLong"] = mapped["ConfirmLong"].fillna(0).to_numpy()
    out["ConfirmShort"] = mapped["ConfirmShort"].fillna(0).to_numpy()
    out["NewSignalBar"] = np.r_[True, label[1:] != label[:-1]].astype(float)
    return out


class RangeReversal1Min(Strategy):
    """
    4時間足の Swing High / Low（compute_signals → map_signals_to_1min の列）で逆張りする。
    entry_mode="limit" はラインに指値、"confirm" は map_confirm_signals_to_1min() のシグナルで成行。
    判定は1本前の4時間足までの確定情報だけを使い、約定は1分足で判定する。
    """
    entry_mode = "limit"
    sl_buffer_atr = 0.5
    tp_range_frac = 0.5
    tp_atr = None
    min_range_atr = 2.0
    breakeven_trigger_r = None
    price_decimals = 5
    risk_pct = 0.02
    filter_mode = None

    def init(self):
        self.upper = self.I(lambda: self.data.SignalSH, overlay=True, name="Range Upper (4H Swing High)")
        self.lower = self.I(lambda: self.data.SignalSL, overlay=True, name="Range Lower (4H Swing Low)")
        self.upper_valid = self.I(lambda: self.data.SignalSHValid, plot=False, overlay=False, name="Upper Valid")
        self.lower_valid = self.I(lambda: self.data.SignalSLValid, plot=False, overlay=False, name="Lower Valid")
        self.atr = self.I(lambda: self.data.SignalATR, overlay=False, name="ATR (Signal, EMA)")
        self.weekly_trend = self.I(lambda: self.data.DowTrend, overlay=False, name="Dow Trend (Weekly)")
        self.h4_trend = self.I(lambda: self.data.H4DowTrend, overlay=False, name="Dow Trend (4H)")
        if self.entry_mode == "confirm":
            self.confirm_long = self.I(lambda: self.data.ConfirmLong, plot=False, overlay=False, name="Confirm Long")
            self.confirm_short = self.I(lambda: self.data.ConfirmShort, plot=False, overlay=False, name="Confirm Short")
            self.new_bar = self.I(lambda: self.data.NewSignalBar, plot=False, overlay=False, name="New 4H Bar")
        self.initial_risk_amount = self.equity * self.risk_pct

    def in_range_regime(self):
        if self.filter_mode is None:
            return True
        trend = self.weekly_trend if self.filter_mode == "weekly" else self.h4_trend
        return trend[-1] == 0

    def place_entry(self, is_long, line, width, atr):
        d = self.price_decimals
        entry = round(line, d)
        tp_dist = width * self.tp_range_frac if self.tp_atr is None else atr * self.tp_atr
        if is_long:
            sl = round(entry - atr * self.sl_buffer_atr, d)
            tp = round(entry + tp_dist, d)
        else:
            sl = round(entry + atr * self.sl_buffer_atr, d)
            tp = round(entry - tp_dist, d)
        sl_dist = abs(entry - sl)
        if sl_dist <= 0 or tp == entry:
            return
        size = int(self.initial_risk_amount / sl_dist)
        if size <= 0:
            return
        if is_long:
            self.buy(size=size, limit=entry, sl=sl, tp=tp)
        else:
            self.sell(size=size, limit=entry, sl=sl, tp=tp)

    def next(self):
        move_sl_to_breakeven(self, self.breakeven_trigger_r)

        # 保有中は反対側のエントリー指値を取り消す（SL/TP の注文は残す）
        if self.position:
            for order in self.orders:
                if not order.is_contingent:
                    order.cancel()
            return

        for order in self.orders:
            order.cancel()

        if self.entry_mode == "confirm":
            self.next_confirm()
            return

        upper, lower, atr = self.upper[-1], self.lower[-1], self.atr[-1]
        if np.isnan(upper) or np.isnan(lower) or np.isnan(atr) or atr <= 0:
            return
        if not (self.upper_valid[-1] and self.lower_valid[-1]):
            return  # どちらかのラインが抜かれている＝レンジが崩れている
        width = upper - lower
        if width <= 0 or width < self.min_range_atr * atr:
            return
        if not self.in_range_regime():
            return

        close = self.data.Close[-1]
        if close < upper:
            self.place_entry(False, upper, width, atr)
        if close > lower:
            self.place_entry(True, lower, width, atr)


    def next_confirm(self):
        if not self.new_bar[-1] or not self.in_range_regime():
            return
        atr = self.atr[-1]  # 反転を確認した足までの ATR
        if np.isnan(atr) or atr <= 0:
            return
        is_long = bool(self.confirm_long[-1])
        if not is_long and not self.confirm_short[-1]:
            return
        d = self.price_decimals
        entry = self.data.Close[-1]  # 目安（実際の約定は次の1分足の始値）
        width = self.upper[-1] - self.lower[-1]
        tp_dist = width * self.tp_range_frac if self.tp_atr is None else atr * self.tp_atr
        sl_dist = atr * self.sl_buffer_atr
        size = int(self.initial_risk_amount / sl_dist) if sl_dist > 0 else 0
        if size <= 0 or tp_dist <= 0:
            return
        if is_long:
            self.buy(size=size, sl=round(entry - sl_dist, d), tp=round(entry + tp_dist, d))
        else:
            self.sell(size=size, sl=round(entry + sl_dist, d), tp=round(entry - tp_dist, d))


def main(cfg=CFG):
    if cfg.execution_timeframe != "1min":
        print('❌ この版は execution_timeframe="1min" のみ対応しています。')
        return
    out_dir = Path(cfg.result_root) / f"{datetime.now():%Y%m%d_%H%M}_{cfg.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol = Path(cfg.data_path).name

    df_1min, avg_relative_spread = load_gmo_click_1min_data(
        cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
    )
    commission_rate = avg_relative_spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * commission_rate, cfg.price_decimals)

    # 週足・4時間足のダウ理論トレンド（全期間で計算してから、検証期間に絞る）
    weekly = resample_to_weekly_bars(df_1min, cfg.session_start_hour)
    weekly_trend, _ = compute_dow_trend(weekly, DowConfig(
        n=cfg.dow_n, atr_period=cfg.dow_atr_period, min_swing_atr=cfg.dow_min_swing_atr,
        use_wick=cfg.dow_use_wick))
    signal_df = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
    h4_trend, _ = compute_dow_trend(signal_df, DowConfig(
        n=cfg.h4_dow_n, atr_period=cfg.h4_dow_atr_period, min_swing_atr=cfg.h4_dow_min_swing_atr,
        use_wick=cfg.h4_dow_use_wick))

    signals = compute_signals(signal_df, window=cfg.window, atr_period=cfg.atr_period,
                              price_decimals=cfg.price_decimals)
    df_ext = map_signals_to_1min(df_1min, signals, cfg.session_start_hour, cfg.signal_hours)
    df_ext = map_weekly_trend_to_1min(df_ext, weekly_trend, cfg.session_start_hour)
    df_ext = map_h4_trend_to_1min(df_ext, h4_trend, cfg.session_start_hour, cfg.signal_hours)
    if cfg.entry_mode == "confirm":
        df_ext = map_confirm_signals_to_1min(df_ext, signal_df, signals, cfg.min_range_atr,
                                             cfg.session_start_hour, cfg.signal_hours)

    trading_day = get_trading_day_label(df_ext.index, cfg.session_start_hour)
    start_ts = pd.Timestamp(cfg.start_date) if cfg.start_date else trading_day.min()
    end_ts = pd.Timestamp(cfg.end_date) if cfg.end_date else trading_day.max()
    df = df_ext.loc[(trading_day >= start_ts) & (trading_day <= end_ts)]
    del df_ext
    if df.empty:
        print("❌ 指定した検証期間にデータがありません。")
        return
    period = f"{start_ts:%Y%m%d}-{end_ts:%Y%m%d}"

    # 検証期間の各週で使われる週足判定（＝前週末の判定）と、レンジ判定だった時間の割合
    used = weekly_trend.shift(1).fillna(0).astype(int)
    labels = {1: "上昇", 0: "レンジ", -1: "下降"}
    used_in_period = used[(used.index >= start_ts) & (used.index <= end_ts + pd.Timedelta(days=6))]
    used_in_period.map(labels).rename("trend").to_csv(out_dir / f"weekly_trend_{period}.csv", encoding="utf-8-sig")
    print("\n================ 検証期間でレンジ判定だった割合（1分足の本数ベース） ================")
    for mode, col in FILTER_COLUMN.items():
        print(f"  {mode:>6}: {(df[col] == 0).mean() * 100:.1f}%")

    params = dict(
        entry_mode=cfg.entry_mode, sl_buffer_atr=cfg.sl_buffer_atr, tp_range_frac=cfg.tp_range_frac,
        tp_atr=cfg.tp_atr, min_range_atr=cfg.min_range_atr,
        breakeven_trigger_r=cfg.breakeven_trigger_r, price_decimals=cfg.price_decimals, risk_pct=cfg.risk_pct,
    )
    rows = {}
    chart_bt = chart_stats = None
    for mode in cfg.compare_modes:
        name = mode or "filter_off"
        print(f"\n================ バックテスト実行中（{symbol} {period} フィルター: {name}） ================")
        bt = Backtest(df, RangeReversal1Min, cash=cfg.cash, commission=commission_func,
                      margin=cfg.margin, exclusive_orders=False)
        stats = add_extra_stats(bt.run(filter_mode=mode, **params))
        rows[name] = summarize(stats, cfg)
        stats["_trades"].to_csv(out_dir / f"trades_{period}_{name}.csv", index=False,
                                float_format=f"%.{cfg.price_decimals}f")
        if mode == cfg.filter_mode:
            chart_bt, chart_stats = bt, stats

    table = pd.DataFrame(rows).T
    print("\n================ 比較（レンジ判定フィルター） ================")
    with pd.option_context("display.unicode.east_asian_width", True,
                           "display.width", 250, "display.max_columns", None):
        print(table.to_string(float_format=lambda v: f"{v:,.2f}", na_rep="-"))
    table.to_csv(out_dir / f"comparison_{period}.csv", encoding="utf-8-sig")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(f"\n💾 比較表・取引履歴・週足トレンドを {out_dir}/ に保存しました")

    if chart_bt is None:
        return out_dir, table
    name = cfg.filter_mode or "filter_off"
    trade_html = make_chart(
        df_1min, out_dir / f"dow_trade_chart_{symbol}_{period}_{name}.html",
        f"{symbol} {start_ts:%Y-%m-%d} ~ {end_ts:%Y-%m-%d} / トレード: 水平ライン逆張り（{name}）",
        start_ts, end_ts, trades=chart_stats["_trades"], weekly_trend=used,
        mode=name, risk=cfg.cash * cfg.risk_pct,
        entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
        session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours,
    )
    # backtesting.py のチャート。表示は判定足にまとめる（main_4H_fixedSL_dow.py と同じ）
    html_path = out_dir / f"chart_{period}_{name}.html"
    chart_df = df.copy()
    chart_df.index = make_chart_index_for_signal_bars(df.index, cfg.session_start_hour, cfg.signal_hours)
    plot_bt = Backtest(chart_df, RangeReversal1Min, cash=cfg.cash, commission=commission_func,
                       margin=cfg.margin, exclusive_orders=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_stats = plot_bt.run(filter_mode=cfg.filter_mode, **params)
    if plot_stats["# Trades"] != chart_stats["# Trades"]:
        print("⚠️ チャート用の再実行結果とトレード数が一致しません。チャートの表示内容にご注意ください。")
    plot_bt.plot(filename=str(html_path), open_browser=False,
                 resample=f"{cfg.signal_hours}h", plot_volume=False)
    append_stats_to_html(html_path, chart_stats, title=f"Stats (range reversal: {name})")
    print(f"📊 チャートを保存しました: {html_path}")
    if OPEN_CHART:
        for p in (trade_html, html_path):
            try:
                webbrowser.open("file://" + os.path.abspath(p), new=2)
            except Exception:
                pass
    return out_dir, table


if __name__ == "__main__":
    main()
