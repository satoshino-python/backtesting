"""
週足ダウ理論フィルター（main_4H_fixedSL_dow.py）＋ 4時間足スイングのトレーリングストップ決済。

【エントリー】main_4H_fixedSL_dow.py と同じ
  - 4時間足の前後 window 本（既定18）の Swing High に買い逆指値 / Swing Low に売り逆指値
  - 週足ダウ理論（n=dow_n）のトレンドと同じ方向だけ（dow_mode="strict": 上昇=買いのみ、下降=売りのみ、レンジ=取引なし）
  - 片方が約定したら反対側のエントリー注文は取り消す

【決済】TP・建値ストップなし。ストップのみ
  - 当初SL: エントリー時点で確定している直近の4時間足 Swing Low（買い）/ Swing High（売り）
    （前後 exit_window 本、既定6）。エントリー価格の反対側に無い場合は発注しない
  - 保有中: 新しく確定した4時間足スイングが現在のSLより有利なら、そこへSLを移す（有利な方向にだけ）

【枚数】当初SLまでの距離で、損失が 初期資金 × risk_pct（1R）になるように決める（複利なし）

【トレンドの質フィルター】（任意。None で無効。すべて前週末／1本前の4時間足で確定した値を使う）
  - min_trend_age : 週足ダウ判定が同じ向きで続いている週数がこれ以上
  - min_weekly_adx: 週足 ADX(14) がこれ以上
  - min_h4_er     : 4時間足の効率比（直近 er_period 本の値動き ÷ 各足の値動きの合計。取引方向を正とする）がこれ以上

結果は result_root/<日時>_<run_label>/ に保存する（前の結果を上書きしない）。
"""
import json
import os
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
    get_latest_swing_high,
    get_latest_swing_low,
    compute_signals,
    map_signals_to_1min,
    add_extra_stats,
)
from main_4H_fixedSL_dow import resample_to_weekly_bars, map_weekly_trend_to_1min, summarize


# ===== 変更するパラメータはここだけ =====
@dataclass(frozen=True)
class SwingExitConfig(Config):
    exit_window: int = 6             # 決済に使う4時間足スイングの前後本数
    dow_n: int = 3                   # 週足ダウ理論のピボット幅（左右 n 週）
    dow_atr_period: int = 14
    dow_min_swing_atr: float = 1.0
    dow_use_wick: bool = True
    dow_mode: str = "strict"         # "strict" / "no_counter" / None（フィルターなし）
    min_trend_age: int = None        # 週足トレンドの継続週数の下限
    min_weekly_adx: float = None     # 週足 ADX(14) の下限
    min_h4_er: float = None          # 4時間足の効率比（取引方向）の下限
    er_period: int = 18              # 効率比の本数（4時間足）
    result_root: str = "dow_results"
    run_label: str = "swingExit"


CFG = SwingExitConfig(
    start_date="2021-01-01",
    end_date="2025-12-31",
    window=18,
    exit_window=6,
    dow_n=3,
    cash=10_000,
    risk_pct=0.02,
)
OPEN_CHART = True


def compute_exit_swings(df_1min, window=6, session_start_hour=17, signal_hours=4):
    """
    判定足（4時間足）の確定済み Swing High / Low（前後 window 本）を1分足に割り当てる（列 ExitSH, ExitSL）。
    ピボットは window 本後の足の確定で判明するため、次の判定足から使う（1本シフト）。
    """
    bars = resample_to_signal_bars(df_1min, session_start_hour=session_start_hour, signal_hours=signal_hours)
    swings = pd.DataFrame({
        "ExitSH": get_latest_swing_high(bars["High"].values, window),
        "ExitSL": get_latest_swing_low(bars["Low"].values, window),
    }, index=bars.index).shift(1)
    mapped = swings.reindex(get_signal_bar_label(df_1min.index, session_start_hour, signal_hours))
    mapped.index = df_1min.index
    return mapped


def weekly_adx(weekly, period=14):
    """週足 ADX（ワイルダー平滑化）"""
    h, l, c = weekly["High"], weekly["Low"], weekly["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), weekly.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), weekly.index)
    ew = lambda x: x.ewm(alpha=1 / period, adjust=False).mean()
    atr = ew(tr)
    pdi, ndi = 100 * ew(pdm) / atr, 100 * ew(ndm) / atr
    return ew(100 * (pdi - ndi).abs() / (pdi + ndi))


def trend_age(trend):
    """同じトレンド判定が続いている本数（その週を含む）"""
    return trend.groupby((trend != trend.shift()).cumsum()).cumcount() + 1


def add_quality_columns(df_ext, df_1min, weekly, weekly_trend, cfg):
    """TrendAge / WeeklyADX（前週末の値）と H4ER（1本前の4時間足の値）を1分足に付ける"""
    from main_4H_fixedSL_dow import week_label
    wk = pd.DataFrame({"TrendAge": trend_age(weekly_trend), "WeeklyADX": weekly_adx(weekly)}).shift(1)
    mapped = wk.reindex(week_label(df_ext.index, cfg.session_start_hour))
    out = df_ext.copy()
    out["TrendAge"] = mapped["TrendAge"].fillna(0).to_numpy()
    out["WeeklyADX"] = mapped["WeeklyADX"].fillna(0).to_numpy()
    bars = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
    c = bars["Close"]
    er = ((c - c.shift(cfg.er_period)) / c.diff().abs().rolling(cfg.er_period).sum()).shift(1)
    out["H4ER"] = er.reindex(get_signal_bar_label(df_ext.index, cfg.session_start_hour, cfg.signal_hours)).fillna(0).to_numpy()
    return out


class SwingBreakoutDowSwingExit1Min(Strategy):
    price_decimals = 5
    risk_pct = 0.02
    dow_mode = "strict"
    min_trend_age = None
    min_weekly_adx = None
    min_h4_er = None

    def init(self):
        self.latest_sh = self.I(lambda: self.data.SignalSH, overlay=True, name="Entry Swing High (4H)")
        self.latest_sl = self.I(lambda: self.data.SignalSL, overlay=True, name="Entry Swing Low (4H)")
        self.sh_valid = self.I(lambda: self.data.SignalSHValid, plot=False, overlay=False, name="SH Valid")
        self.sl_valid = self.I(lambda: self.data.SignalSLValid, plot=False, overlay=False, name="SL Valid")
        self.exit_sh = self.I(lambda: self.data.ExitSH, overlay=True, name="Exit Swing High (4H)")
        self.exit_sl = self.I(lambda: self.data.ExitSL, overlay=True, name="Exit Swing Low (4H)")
        self.dow_trend = self.I(lambda: self.data.DowTrend, overlay=False, name="Dow Trend (Weekly)")
        self.initial_risk_amount = self.equity * self.risk_pct

    def quality_ok(self, is_long):
        if self.min_trend_age is not None and self.data.TrendAge[-1] < self.min_trend_age:
            return False
        if self.min_weekly_adx is not None and self.data.WeeklyADX[-1] < self.min_weekly_adx:
            return False
        if self.min_h4_er is not None:
            er = self.data.H4ER[-1] if is_long else -self.data.H4ER[-1]
            if er < self.min_h4_er:
                return False
        return True

    def allowed(self, is_long):
        if not self.quality_ok(is_long):
            return False
        trend = self.dow_trend[-1]
        if self.dow_mode is None:
            return True
        if self.dow_mode == "strict":
            return trend == 1 if is_long else trend == -1
        return trend != -1 if is_long else trend != 1  # "no_counter"

    def trail_stops(self):
        close = self.data.Close[-1]
        for trade in self.trades:
            new_sl = self.exit_sl[-1] if trade.is_long else self.exit_sh[-1]
            if np.isnan(new_sl):
                continue
            new_sl = round(new_sl, self.price_decimals)
            if trade.sl is not None and ((new_sl <= trade.sl) if trade.is_long else (new_sl >= trade.sl)):
                continue
            if (new_sl < close) if trade.is_long else (new_sl > close):
                trade.sl = new_sl
            else:
                trade.close()  # 新しいスイングが既に現在値の反対側（ほぼ起きない）→ 成行で決済

    def place_entry(self, is_long, entry_price):
        sl = self.exit_sl[-1] if is_long else self.exit_sh[-1]
        if np.isnan(sl):
            return
        sl = round(sl, self.price_decimals)
        if (sl >= entry_price) if is_long else (sl <= entry_price):
            return  # 直近スイングがエントリー価格の反対側に無い
        size = int(self.initial_risk_amount / abs(entry_price - sl))
        if size <= 0:
            return
        if is_long:
            self.buy(size=size, stop=entry_price, sl=sl)
        else:
            self.sell(size=size, stop=entry_price, sl=sl)

    def next(self):
        self.trail_stops()
        if self.position:
            for order in self.orders:
                if not order.is_contingent:
                    order.cancel()
            return

        for order in self.orders:
            order.cancel()
        current_sh, current_sl = self.latest_sh[-1], self.latest_sl[-1]
        close = self.data.Close[-1]
        if not np.isnan(current_sh) and close < current_sh and self.sh_valid[-1] and self.allowed(True):
            self.place_entry(True, round(current_sh, self.price_decimals))
        if not np.isnan(current_sl) and close > current_sl and self.sl_valid[-1] and self.allowed(False):
            self.place_entry(False, round(current_sl, self.price_decimals))


def prepare(cfg):
    """データを読み込み、検証期間の1分足（シグナル・決済スイング・週足トレンド・質の列付き）を作る"""
    df_1min, avg_relative_spread = load_gmo_click_1min_data(
        cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
    )
    commission_rate = avg_relative_spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * commission_rate, cfg.price_decimals)

    weekly = resample_to_weekly_bars(df_1min, cfg.session_start_hour)
    weekly_trend, _ = compute_dow_trend(weekly, DowConfig(
        n=cfg.dow_n, atr_period=cfg.dow_atr_period, min_swing_atr=cfg.dow_min_swing_atr,
        use_wick=cfg.dow_use_wick))

    signal_df = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
    signals = compute_signals(signal_df, window=cfg.window, atr_period=cfg.atr_period,
                              price_decimals=cfg.price_decimals)
    df_ext = map_signals_to_1min(df_1min, signals, cfg.session_start_hour, cfg.signal_hours)
    df_ext = df_ext.join(compute_exit_swings(df_1min, cfg.exit_window, cfg.session_start_hour, cfg.signal_hours))
    df_ext = map_weekly_trend_to_1min(df_ext, weekly_trend, cfg.session_start_hour)
    df_ext = add_quality_columns(df_ext, df_1min, weekly, weekly_trend, cfg)

    trading_day = get_trading_day_label(df_ext.index, cfg.session_start_hour)
    start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
    df = df_ext.loc[(trading_day >= start_ts) & (trading_day <= end_ts)]
    del df_ext
    return df_1min, df, commission_func, weekly_trend


def run_backtest(df, cfg, commission_func):
    bt = Backtest(df, SwingBreakoutDowSwingExit1Min, cash=cfg.cash, commission=commission_func,
                  margin=cfg.margin, exclusive_orders=False)
    stats = add_extra_stats(bt.run(
        dow_mode=cfg.dow_mode, price_decimals=cfg.price_decimals, risk_pct=cfg.risk_pct,
        min_trend_age=cfg.min_trend_age, min_weekly_adx=cfg.min_weekly_adx, min_h4_er=cfg.min_h4_er))
    return bt, stats


def main(cfg=CFG):
    out_dir = Path(cfg.result_root) / f"{datetime.now():%Y%m%d_%H%M}_{cfg.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    symbol = Path(cfg.data_path).name
    df_1min, df, commission_func, weekly_trend = prepare(cfg)
    start_ts, end_ts = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
    period = f"{start_ts:%Y%m%d}-{end_ts:%Y%m%d}"

    used = weekly_trend.shift(1).fillna(0).astype(int)
    used_in_period = used[(used.index >= start_ts) & (used.index <= end_ts + pd.Timedelta(days=6))]
    labels = {1: "上昇", 0: "レンジ", -1: "下降"}
    used_in_period.map(labels).rename("trend").to_csv(out_dir / f"weekly_trend_{period}.csv", encoding="utf-8-sig")

    print(f"\n================ バックテスト実行中（{symbol} {period} dow={cfg.dow_mode}） ================")
    bt, stats = run_backtest(df, cfg, commission_func)
    print(stats)
    trades = stats["_trades"]
    trades.to_csv(out_dir / f"trades_{period}.csv", index=False, float_format=f"%.{cfg.price_decimals}f")
    summary = summarize(stats, cfg)
    pd.Series(summary).to_csv(out_dir / "summary.csv", encoding="utf-8-sig")
    stats.drop(["_strategy", "_equity_curve", "_trades"]).to_csv(out_dir / "stats.csv", encoding="utf-8-sig")
    stats["_equity_curve"]["Equity"].resample("1D").last().dropna().to_csv(out_dir / "equity_daily.csv")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    print(pd.Series(summary).to_string())

    trade_html = make_chart(
        df_1min, out_dir / f"dow_trade_chart_{symbol}_{period}_{cfg.dow_mode}.html",
        f"{symbol} {start_ts:%Y-%m-%d} ~ {end_ts:%Y-%m-%d} / トレード: {cfg.dow_mode}・4H n={cfg.exit_window} スイング決済",
        start_ts, end_ts, trades=trades, weekly_trend=used,
        mode=cfg.dow_mode, risk=cfg.cash * cfg.risk_pct,
        entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
        session_start_hour=cfg.session_start_hour, signal_hours=cfg.signal_hours,
    )
    html_path = out_dir / f"chart_{period}.html"
    bt.plot(filename=str(html_path), open_browser=False, resample=f"{cfg.signal_hours}h", plot_volume=False)
    print(f"💾 結果を {out_dir}/ に保存しました")
    if OPEN_CHART:
        for p in (trade_html, html_path):
            try:
                webbrowser.open("file://" + os.path.abspath(p), new=2)
            except Exception:
                pass
    return out_dir, stats


if __name__ == "__main__":
    main()
