"""
20MAの傾きの効果だけを取り出す検証。trend_er_split.py の「MAの並びだけの全取引」に、エントリー時（前日確定）の
日足 20MA の傾き（k=1,3 日）が取引方向と一致していたかを付ける。結果は同じ実行フォルダの slope_split.csv に保存する。
使い方: python trend_slope_split.py <er_split の実行フォルダ>
"""
import contextlib, io, sys, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import load_gmo_click_1min_data, resample_to_signal_bars, get_signal_bar_label
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for


def one(args):
    pair, t = args
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, _ = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                         month_range=zip_month_range(cfg), spread_period=(s, e), session_start_hour=cfg.session_start_hour)
        d1 = resample_to_signal_bars(df, cfg.session_start_hour, 24)
        ma = d1["Close"].rolling(20).mean()
        f = pd.DataFrame({f"d{k}": ma - ma.shift(k) for k in (1, 3)}).shift(1).reindex(get_signal_bar_label(df.index, cfg.session_start_hour, 24))
        f.index = df.index
        et = pd.DatetimeIndex(pd.to_datetime(t["EntryTime"], utc=True)).tz_convert(df.index.tz)
        pos = df.index.get_indexer(et)
    t = t.copy(); sign = np.where(t["Side"] == "L", 1.0, -1.0)
    for k in (1, 3):
        t[f"slope{k}"] = f[f"d{k}"].to_numpy()[pos] * sign   # 取引方向にプラス=傾きが取引方向と一致
    return t


if __name__ == "__main__":
    d = Path(sys.argv[1]); t = pd.read_csv(d / "trades_ma_only.csv")
    with ProcessPoolExecutor(m.WORKERS) as ex:
        out = pd.concat(list(ex.map(one, [(p, g) for p, g in t.groupby("Pair")])), ignore_index=True)
    out.to_csv(d / "slope_split.csv", index=False, encoding="utf-8-sig")
    print(len(out), out[["slope1", "slope3"]].isna().sum().tolist(), round((out.slope1 > 0).mean(), 3))
