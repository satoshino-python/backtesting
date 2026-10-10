"""
GMOクリック証券の1分足（histData/<ペア>/*.zip）から、カップウィズハンドル用の足を作る。

- 読み込みは既存の main_4H_fixedSL.load_gmo_click_1min_data() を使う（日本時間 → NY時間の変換、_EX 系列の除外を含む）。
- 足の区切りは NY 17:00（NYクローズ）。日足のラベルは取引日（NY 17:00〜翌17:00 の大半が含まれる側の日付、月〜金）。
- 価格は仕様どおり BID 基準。スプレッドは costs.py で別途加味する。
- 1分足の読み込みは1ペア数十秒かかるため、cache/ に pickle で保存する（.gitignore 済み。消せば作り直す）。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (
    load_gmo_click_1min_data,
    resample_to_signal_bars,
    signal_bar_trading_day,
)

DATA_ROOT = Path("histData")
CACHE_DIR = Path("cache")
FX_PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "USDCHF", "USDJPY")


def price_decimals_for(pair):
    return 3 if pair.upper().endswith("JPY") else 5


def pip_size_for(pair):
    """円ペアは 0.01、それ以外は 0.0001（仕様 7）"""
    return 0.01 if pair.upper().endswith("JPY") else 0.0001


def available_fx_pairs(data_root=DATA_ROOT):
    """histData/ にある6文字の通貨ペアのフォルダ（SP500 などの FX 以外は除く）"""
    return tuple(sorted(p.name for p in Path(data_root).iterdir()
                        if p.is_dir() and len(p.name) == 6 and p.name.isalpha() and any(p.glob("*.zip"))))


def load_1min(pair, price_side="bid", data_root=DATA_ROOT, use_cache=True):
    """1分足（NY時間、Open/High/Low/Close/Volume）と、データから測った ASK−BID の統計（pips）を返す"""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"m1_{pair}_{price_side}.pkl"
    if use_cache and cache.exists():
        obj = pd.read_pickle(cache)
        return obj["df"], obj["spread"]
    dec = price_decimals_for(pair)
    df, rel_spread = load_gmo_click_1min_data(Path(data_root) / pair, price_side=price_side, price_decimals=dec)
    spread = {"mean_relative": rel_spread, "mean_pips": rel_spread * float(df["Close"].median()) / pip_size_for(pair)}
    pd.to_pickle({"df": df, "spread": spread}, cache)
    return df, spread


def to_bars(df_1min, bar_hours=24, session_start_hour=17, min_minutes=60):
    """
    1分足を bar_hours 時間足（NY session_start_hour 時起点）にまとめる。
    日足（24）のときの index は取引日（タイムゾーンなし）。それ以外は足の開始時刻（NY現地時刻）。
    週末の数分だけの足（金曜17時直後に残る端数など）は min_minutes 未満なら捨てる。
    """
    bars = resample_to_signal_bars(df_1min, session_start_hour, bar_hours)
    label = pd.Series(df_1min.index.tz_localize(None), index=df_1min.index)
    counts = df_1min.groupby(
        ((label - pd.Timedelta(hours=session_start_hour)).dt.floor(f"{bar_hours}h")
         + pd.Timedelta(hours=session_start_hour)).to_numpy()
    ).size()
    bars = bars[counts.reindex(bars.index).fillna(0).to_numpy() >= min_minutes]
    if bar_hours == 24:
        bars.index = signal_bar_trading_day(bars.index, session_start_hour)
        bars.index.name = "Date"
    bars = bars[["Open", "High", "Low", "Close"]].astype(float)
    bars.columns = [c.lower() for c in bars.columns]
    return bars


def load_bars(pair, bar_hours=24, price_side="bid", data_root=DATA_ROOT, use_cache=True):
    """取引に使う足（open/high/low/close、BID）を返す。日足は cache/d1_<ペア>_<side>.pkl に保存する"""
    CACHE_DIR.mkdir(exist_ok=True)
    cache = CACHE_DIR / f"bars{bar_hours}h_{pair}_{price_side}.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    df, spread = load_1min(pair, price_side, data_root, use_cache)
    bars = to_bars(df, bar_hours)
    bars.attrs["spread"] = spread
    pd.to_pickle(bars, cache)
    return bars
