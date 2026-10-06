"""
fast_engine.run_fast() が backtesting.py（SwingBreakoutStrategy1Min）と同じ結果になるかを確かめる。

同じデータ・同じ設定で両方を実行し、取引履歴を1件ずつ照らし合わせ、主要な統計値と所要時間を比べる。
一致しなかった場合は両方の取引履歴を fast_compare_results/<日時>_<ペア>/ に保存する。

例: python compare_fast_engine.py EURUSD 2024-01-01 2024-12-31
    python compare_fast_engine.py EURUSD 2024-01-01 2024-12-31 breakeven_trigger_r=1.0
    python compare_fast_engine.py EURUSD 2024-01-01 2024-12-31 exit_mode=swing_trail initial_sl_rule=near
"""
import sys
import time
import dataclasses
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from backtesting import Backtest

from main_4H_fixedSL import (
    load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
    map_signals_to_1min, get_trading_day_label, SwingBreakoutStrategy1Min,
)
from main_4H_fixedSL_multi import (
    BASE_CFG, DATA_ROOT, zip_month_range, price_decimals_for, compute_h1_swings, SwingBreakoutTrail1Min,
)
from fast_engine import run_fast

TRADE_COLS = ["Size", "EntryBar", "ExitBar", "EntryPrice", "ExitPrice", "SL", "TP",
              "PnL", "Commission", "ReturnPct", "EntryTime", "ExitTime"]
STAT_KEYS = ["Equity Final [$]", "Return [%]", "Max. Drawdown [%]", "# Trades", "Win Rate [%]",
             "Profit Factor", "Sharpe Ratio", "Exposure Time [%]", "Buy & Hold Return [%]"]


def prepare(pair, cfg, h1_window=None):
    s = pd.Timestamp(cfg.start_date)
    e = pd.Timestamp(cfg.end_date)
    df_1min, spread = load_gmo_click_1min_data(
        cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
        month_range=zip_month_range(cfg), spread_period=(s, e),
        session_start_hour=cfg.session_start_hour)
    signal_df = resample_to_signal_bars(df_1min, cfg.session_start_hour, cfg.signal_hours)
    signals = compute_signals(signal_df, cfg.window, cfg.atr_period, cfg.price_decimals)
    ext = map_signals_to_1min(df_1min, signals, cfg.session_start_hour, cfg.signal_hours)
    if h1_window is not None:  # swing_trail 用の1時間足スイング
        ext = ext.join(compute_h1_swings(df_1min, h1_window, cfg.session_start_hour))
    day = get_trading_day_label(ext.index, cfg.session_start_hour)
    df = ext.loc[(day >= s) & (day <= e)]
    rate = spread / 2

    def commission_func(order_size, price):
        return round(abs(order_size) * price * rate, cfg.price_decimals)

    return df, commission_func


def compare(pair, start, end, exit_mode="fixed", initial_sl_rule="atr", h1_window=5, **overrides):
    """
    exit_mode="fixed" は SwingBreakoutStrategy1Min、"swing_trail" は SwingBreakoutTrail1Min と比べる
    （h1_window: 1時間足スイングの前後本数、initial_sl_rule: 当初SLの決め方）。
    overrides で BASE_CFG の項目を上書きできる（例: breakeven_trigger_r=1.0）
    """
    cfg = dataclasses.replace(BASE_CFG, data_path=Path(DATA_ROOT) / pair, start_date=start,
                              end_date=end, price_decimals=price_decimals_for(pair), **overrides)
    trail = exit_mode == "swing_trail"
    df, commission_func = prepare(pair, cfg, h1_window if trail else None)
    params = dict(sl_atr_multiplier=cfg.sl_atr_multiplier, price_decimals=cfg.price_decimals,
                  risk_pct=cfg.risk_pct)
    if trail:
        strategy = SwingBreakoutTrail1Min
        params.update(initial_sl_rule=initial_sl_rule)
        fast_params = dict(params, exit_mode="swing_trail")
        label = f"swing_trail（H1 window={h1_window}, 当初SL={initial_sl_rule}）"
    else:
        strategy = SwingBreakoutStrategy1Min
        params.update(tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=cfg.breakeven_trigger_r)
        fast_params = params
        label = f"fixed（TP {cfg.tp_atr_multiplier}, 建値ストップ {cfg.breakeven_trigger_r}）"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        bt = Backtest(df, strategy, cash=cfg.cash, commission=commission_func,
                      margin=cfg.margin, exclusive_orders=False)
        ref = bt.run(**params)
        t_ref = time.perf_counter() - t

        t = time.perf_counter()
        fast = run_fast(df, cash=cfg.cash, commission=commission_func, margin=cfg.margin, **fast_params)
        t_fast = time.perf_counter() - t

    a = ref["_trades"][TRADE_COLS].reset_index(drop=True)
    b = fast["_trades"][TRADE_COLS].reset_index(drop=True)
    ok = len(a) == len(b)
    first_diff = None
    if ok:
        for col in TRADE_COLS:
            x, y = a[col], b[col]
            if pd.api.types.is_numeric_dtype(x):
                same = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
            else:
                same = ((x == y) | (x.isna() & y.isna())).to_numpy()
            if not same.all():
                ok = False
                row = int(np.flatnonzero(~same)[0])
                first_diff = row if first_diff is None else min(first_diff, row)

    eq_diff = float(np.max(np.abs(ref["_equity_curve"]["Equity"].to_numpy()
                                  - fast["_equity_curve"]["Equity"].to_numpy())))

    print(f"\n===== {pair} {start} ~ {end}（1分足 {len(df):,} 本） {label} {overrides or ''} =====")
    print(f"backtesting.py: {t_ref:7.2f} 秒 / fast_engine: {t_fast:6.2f} 秒（{t_ref / t_fast:,.0f} 倍）")
    print(f"取引履歴: {'一致' if ok else '不一致'}（{len(a)} 件 / {len(b)} 件）  資産曲線の最大差: {eq_diff:.3g}")
    for k in STAT_KEYS:
        print(f"  {k:24s} {ref[k]!s:>24} {fast[k]!s:>24}")

    if not ok:
        out = Path("fast_compare_results") / f"{datetime.now():%Y%m%d_%H%M%S}_{pair}"
        out.mkdir(parents=True, exist_ok=True)
        a.to_csv(out / "trades_backtesting.csv", index=False)
        b.to_csv(out / "trades_fast.csv", index=False)
        if first_diff is not None:
            print("最初に食い違った取引:")
            print(pd.concat({"backtesting": a.iloc[first_diff], "fast": b.iloc[first_diff]}, axis=1))
        print(f"取引履歴を保存しました: {out}")
    return dict(pair=pair, start=start, end=end, exit_mode=exit_mode, bars=len(df), trades=len(a), match=ok,
                equity_max_diff=eq_diff, sec_backtesting=t_ref, sec_fast=t_fast)


if __name__ == "__main__":
    pair = sys.argv[1] if len(sys.argv) > 1 else "EURUSD"
    start = sys.argv[2] if len(sys.argv) > 2 else "2024-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2024-03-31"
    # 4番目以降の引数 "名前=値" で compare() の引数・BASE_CFG を上書き（値は数値・None・文字列）
    overrides = {}
    for arg in sys.argv[4:]:
        k, v = arg.split("=", 1)
        if v == "None":
            overrides[k] = None
        else:
            try:
                overrides[k] = int(v) if k == "h1_window" else float(v)
            except ValueError:
                overrides[k] = v
    compare(pair, start, end, **overrides)
