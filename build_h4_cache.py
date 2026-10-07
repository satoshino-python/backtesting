"""
1分足（histData/<ペア>）を NY 17:00 区切りの4時間足にまとめて cache/h4_<ペア>.pkl に保存する。
trend_filter_study.py で日足・週足の指標を計算するための下準備（1分足を毎回読み込まないため）。
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from main_4H_fixedSL import load_gmo_click_1min_data, resample_to_signal_bars

PAIRS = ("AUDUSD", "EURUSD", "GBPUSD", "SP500", "USDCHF", "USDJPY")
CACHE = Path("cache")


def build(pair):
    out = CACHE / f"h4_{pair}.pkl"
    if out.exists():
        return pair, "skip"
    dec = 3 if pair.endswith("JPY") else (2 if pair in ("SP500", "US500") else 5)
    df, _ = load_gmo_click_1min_data(f"histData/{pair}", price_decimals=dec, month_range=(202001, 202512))
    bars = resample_to_signal_bars(df, 17, 4)
    bars.to_pickle(out)
    return pair, len(bars)


if __name__ == "__main__":
    CACHE.mkdir(exist_ok=True)
    pairs = sys.argv[1:] or PAIRS
    with ProcessPoolExecutor(2) as ex:
        for pair, n in ex.map(build, pairs):
            print(pair, n, flush=True)
