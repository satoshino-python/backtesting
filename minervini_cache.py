"""
ミネルヴィニFX検証用の1分足キャッシュ（cache/min_<ペア>.npz）を作る。

load_gmo_click_1min_data で読んだ仲値の1分足を numpy 配列にして保存する（ZIP の読み込みが約10秒/ペアかかるため）。
  ts: 時刻（UTC ns）, o/h/l/c: 仲値, day: 取引日ラベル（NY17:00区切り、datetime64[D] を int64 日数に）, spread: 平均相対スプレッド
cache/ は .gitignore 済み。python minervini_cache.py で全ペア作り直す。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import load_gmo_click_1min_data, get_trading_day_label
from main_4H_fixedSL_multi import price_decimals_for

DATA_ROOT = Path("histData")
CACHE_DIR = Path("cache")
FX_PAIRS = ["AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPAUD", "GBPCHF", "GBPJPY",
            "GBPUSD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY"]


def cache_path(pair):
    return CACHE_DIR / f"min_{pair}.npz"


def build_one(pair):
    df, spread = load_gmo_click_1min_data(DATA_ROOT / pair, price_side="mid", price_decimals=price_decimals_for(pair))
    df = df[~df.index.duplicated(keep="first")].sort_index()
    day = get_trading_day_label(df.index)
    CACHE_DIR.mkdir(exist_ok=True)
    np.savez(cache_path(pair), ts=df.index.tz_convert("UTC").as_unit("ns").asi8, o=df["Open"].to_numpy(float), h=df["High"].to_numpy(float),
             l=df["Low"].to_numpy(float), c=df["Close"].to_numpy(float),
             day=day.values.astype("datetime64[D]").astype("int64"), spread=np.float64(spread))
    return pair, len(df)


if __name__ == "__main__":
    pairs = sys.argv[1:] or FX_PAIRS
    with ProcessPoolExecutor(4) as ex:
        for pair, n in ex.map(build_one, pairs):
            print(pair, n, flush=True)
