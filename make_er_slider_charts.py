"""
ER の期間・しきい値をスライダーで動かせるトレード付きチャートを作る。
トレードは「日足MAの並び（20>75>200 / 逆）だけで入った全取引」（ERでは絞らない）。チャートのスライダーで ER 期間としきい値を
動かすと、ER フィルターを通る取引は濃く、除外される取引は薄く表示され、通過/除外それぞれの件数・平均R・合計Rが更新される。
日足MA・ER・背景は、前日に確定した日足の値（検証と同じ）をページ側で計算する。
使い方: python make_er_slider_charts.py [ペア ...]   → trend_results/<日時>_er_slider_charts/
"""
import contextlib, io, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, map_signals_to_1min,
                             get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for
from fast_engine import run_fast
from dow_swing_chart import make_chart


def one(pair, out):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, spread = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                              month_range=zip_month_range(cfg), spread_period=(s, e),
                                              session_start_hour=cfg.session_start_hour)
        h4 = resample_to_signal_bars(df, cfg.session_start_hour, 4)
        d1 = resample_to_signal_bars(df, cfg.session_start_hour, 24)
        sig = m.breakout_signals(h4, cfg.window, cfg.price_decimals)
        base = map_signals_to_1min(df, sig, cfg.session_start_hour, 4)
        lab = get_signal_bar_label(df.index, cfg.session_start_hour, 24)
        day = get_trading_day_label(df.index, cfg.session_start_hour)
        inp = np.asarray((day >= s) & (day <= e))
        c = d1["Close"]
        ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS)
        f = pd.DataFrame({"up": (ma_s > ma_m) & (ma_m > ma_l), "down": (ma_s < ma_m) & (ma_m < ma_l), "ok": ma_l.notna()}).shift(1).reindex(lab)
        ok = f["ok"].fillna(False).to_numpy(bool)
        d = base.copy(); d["AllowLong"], d["AllowShort"] = f["up"].fillna(False).to_numpy(bool) & ok, f["down"].fillna(False).to_numpy(bool) & ok
        d = d.loc[inp]
        rate = spread / 2; risk = cfg.cash * cfg.risk_pct
        comm = lambda size, price: round(abs(size) * price * rate, cfg.price_decimals)  # noqa: E731
        st = run_fast(d, cash=cfg.cash, commission=comm, margin=cfg.margin, sl_atr_multiplier=cfg.sl_atr_multiplier,
                      tp_atr_multiplier=cfg.tp_atr_multiplier, breakeven_trigger_r=None, price_decimals=cfg.price_decimals,
                      risk_pct=cfg.risk_pct, exit_mode="fixed")
        fd, ld = day[inp][[0, -1]]
        make_chart(df, out / f"er_slider_chart_{pair}.html",
                   f"{pair} {fd:%Y-%m-%d} ~ {ld:%Y-%m-%d} / 日足MAの並びだけで入った全取引 + ERスライダー",
                   fd, ld, trades=st["_trades"], weekly_trend=None, mode="fixed", risk=risk,
                   entry_window=cfg.window, entry_atr=14, price_decimals=cfg.price_decimals,
                   session_start_hour=cfg.session_start_hour, signal_hours=4,
                   daily_ma=m.MA_PERIODS, daily_er=(20, m.ER_THRESHOLD), live_daily=True,
                   entry_signals=sig, timeframes=("4H", "D1", "W1"))
        tr = st["_trades"]; r = (tr["PnL"] / risk).sum()
    return pair, len(tr), float(r)


if __name__ == "__main__":
    pairs = sys.argv[1:] or list(m.PAIRS)
    out = m.RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_er_slider_charts"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    with ProcessPoolExecutor(m.WORKERS) as ex:
        for p, n, r in ex.map(one, pairs, [out] * len(pairs)):
            print("ok", p, n, round(r, 1), flush=True); rows.append(dict(Pair=p, Trades=n, TotalR=round(r, 2)))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print("saved", out)
