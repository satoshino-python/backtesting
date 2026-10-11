"""trend_results/<実行>/ の trades_all.csv（spec）から、トレード付きチャートだけを作り直す（日足MA・日足ERを付けた版）。
使い方: python regen_trend_charts.py <実行フォルダ> [ペア ...]"""
import contextlib, io, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
from main_4H_fixedSL import (load_gmo_click_1min_data, get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from dow_swing_chart import make_chart
import main_4H_trend_breakout_fast as m


def one(pair, out_dir):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, _ = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                         month_range=zip_month_range(cfg), spread_period=(s, e),
                                         session_start_hour=cfg.session_start_hour)
        from main_4H_fixedSL import resample_to_signal_bars
        d1 = resample_to_signal_bars(df, cfg.session_start_hour, 24)
        f = m.daily_filter(d1).shift(1).reindex(get_signal_bar_label(df.index, cfg.session_start_hour, 24))
        er = f["ER"].to_numpy(float)
        ok = ~np.isnan(er) & ~np.isnan(f["up"].astype(float).to_numpy())
        strong = ok & (er >= m.ER_THRESHOLD)
        al, as_ = f["up"].fillna(False).to_numpy(bool) & strong, f["down"].fillna(False).to_numpy(bool) & strong
        state = pd.Series(np.where(al & as_, 2, np.where(al, 1, np.where(as_, -1, 0))), index=df.index)
        day = get_trading_day_label(df.index, cfg.session_start_hour)
        fd, ld = day[(day >= s) & (day <= e)][[0, -1]]
        t = pd.read_csv(out_dir / "trades_all.csv").query("Variant=='spec' and Pair==@pair").reset_index(drop=True)
        make_chart(df, out_dir / f"dow_trade_chart_{pair}.html",
                   f"{pair} {fd:%Y-%m-%d} ~ {ld:%Y-%m-%d} / 日足MA20>75>200+ER(20)≥0.3 → 4H 18本ブレイク SL1.0/TP1.5 ATR",
                   fd, ld, trades=t, weekly_trend=None, mode="fixed", risk=cfg.cash * cfg.risk_pct,
                   entry_window=cfg.window, entry_atr=14, price_decimals=cfg.price_decimals,
                   session_start_hour=cfg.session_start_hour, signal_hours=4, filter_state=state,
                   filter_label="日足 MA20/75/200 並び + ER(20)≥0.3（前日確定）",
                   daily_ma=m.MA_PERIODS, daily_er=(m.ER_PERIOD, m.ER_THRESHOLD))
    return pair


if __name__ == "__main__":
    out = Path(sys.argv[1]); pairs = sys.argv[2:] or list(m.PAIRS)
    with ProcessPoolExecutor(4) as ex:
        for p in ex.map(one, pairs, [out] * len(pairs)):
            print("ok", p, flush=True)
