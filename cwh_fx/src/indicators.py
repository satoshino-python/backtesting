"""指標（仕様 3）: True Range、ATR（ワイルダー平滑）、SMA、スイングハイ/ロー。すべて足 t までの値だけで計算する。"""
import numpy as np
import pandas as pd


def true_range(high, low, close):
    """TR_t = max(H−L, |H−C_{t-1}|, |L−C_{t-1}|)。最初の足は H−L"""
    high, low, close = (np.asarray(x, dtype=float) for x in (high, low, close))
    prev = np.r_[np.nan, close[:-1]]
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    tr[0] = high[0] - low[0]
    return tr


def atr_wilder(high, low, close, period):
    """ワイルダー平滑の ATR。最初の値は TR の最初の period 本の単純平均（それより前は NaN）"""
    tr = true_range(high, low, close)
    out = np.full(len(tr), np.nan)
    if len(tr) < period:
        return out
    out[period - 1] = tr[:period].mean()
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def sma(values, period):
    return pd.Series(np.asarray(values, dtype=float)).rolling(period, min_periods=period).mean().to_numpy()


def swing_high_mask(high, window=5):
    """
    スイングハイ: 左右 window 本の中で最高値（同値を含む）の足。
    足 i のスイングは足 i+window の確定で初めて分かる。右側が window 本そろわない末尾と、
    左側がそろわない先頭は False。使う側は「i + window <= t」の足だけを使うこと。
    """
    h = pd.Series(np.asarray(high, dtype=float))
    roll = h.rolling(2 * window + 1, center=True, min_periods=2 * window + 1).max()
    return (h >= roll).to_numpy() & roll.notna().to_numpy()


def swing_low_mask(low, window=5):
    return swing_high_mask(-np.asarray(low, dtype=float), window)
