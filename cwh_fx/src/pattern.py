"""
カップウィズハンドルの検出（仕様 4）。

detect_cup_with_handle(df, params, t) は「足 t の確定時点で、ハンドルが t で終わるセットアップが成立しているか」を返す。
df.iloc[:t+1] 以外は参照しない（指標はすべて過去の値だけで計算し、スイングは i+window <= t のものだけを使う）。

ショート（逆カップ）は価格を符号反転した系列に同じロジックを適用する（仕様 8）。返す Setup の価格は元の符号に戻した値で、
ショートでは h_left = 左縁の安値、l_cup = カップの天井、h_right = 右縁の安値（ピボット）、l_handle = ハンドルの高値 になる。

仕様に明記がなく、こちらで決めた解釈（cwh_fx/README.md にも記載）:
  - 右縁: 既定（right_rim_mode="min"）は仕様 4.3 の文面どおり H_right >= H_left − 1.0×ATR50 の下限だけ。
    パラメータの説明「右縁は左縁高値からこの範囲内」を上下の範囲と読む場合は "range"（|H_right − H_left| <= 1.0×ATR50）。
  - 左縁は「カップの縁」なので、左縁からカップの底までの間に左縁より高い足が無いこと（left_rim_highest=True）。
  - ATR50 は、左縁・カップに関する値（事前上昇・深さ・丸み・右縁）は ATR50_{i_left}、ハンドルの値（押し幅・ボラ収縮）は ATR50_t。
  - 1つの t で成立する左縁の候補が複数ある場合は、いちばん新しい（t に近い）左縁を採用する。
  - カップの底 i_cup = i_left〜t の最安値の位置（ハンドル安値はカップ上半分にあるので、i_left〜i_right の最安値と同じになる）。
    右縁 i_right = i_cup〜t の最高値の位置（同値は最初の足）。ハンドル = i_right の翌足〜t。
"""
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .indicators import atr_wilder, sma, swing_high_mask


@dataclass
class Setup:
    i_left: int
    i_cup: int
    i_right: int
    i_handle_end: int
    h_left: float
    l_cup: float
    h_right: float
    l_handle: float
    depth_atr: float
    pullback_atr: float
    direction: int = 1          # 1=ロング（カップ） / -1=ショート（逆カップ）
    cup_len: int = 0
    handle_len: int = 0
    round_ratio: float = 0.0
    prior_rise_atr: float = 0.0

    @property
    def key(self):
        """同じ形（同じ左縁・右縁）を見分けるキー"""
        return (self.direction, self.i_left, self.i_right)

    @property
    def pivot(self):
        return self.h_right

    def to_dict(self):
        return asdict(self)


def prepare(df, params, direction=1):
    """
    検出に使う配列をまとめて計算する（各値は足 t までのデータだけで決まる）。
    direction=-1 では価格を符号反転する（high' = −low, low' = −high）。
    """
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    atr14 = atr_wilder(h, l, c, params["atr_short"])
    atr50 = atr_wilder(h, l, c, params["atr_long"])
    if direction == -1:
        o, h, l, c = -o, -l, -h, -c
    return dict(
        direction=direction, open=o, high=h, low=l, close=c, atr14=atr14, atr50=atr50,
        sma=sma(c, params["sma_trend"]),
        swing=swing_high_mask(h, params.get("swing_window", 5)),
    )


def trend_ok(ind, params, t):
    """仕様 4.1: close_t > SMA200_t かつ SMA200_t > SMA200_{t−20}（反転系列ではショートの条件になる）"""
    if not params.get("use_trend_filter", True):
        return True
    lb = params["sma_slope_lookback"]
    if t < lb:
        return False
    s, s0 = ind["sma"][t], ind["sma"][t - lb]
    return bool(np.isfinite(s) and np.isfinite(s0) and ind["close"][t] > s and s > s0)


# 判定の段階（diagnostics.py の集計で、どこで落ちたかを示すのに使う）
STAGES = ("ATR", "事前上昇", "カップの底", "ハンドル長", "カップ長", "左縁が最高値", "深さ", "底の位置", "丸み",
          "右縁(下限)", "右縁(上限)", "押し幅", "押し/深さ", "ハンドル位置", "ATR収縮", "ハンドル中の終値")


def _check_left(ind, params, t, i_left, trace=False):
    """
    左縁 i_left を仮定してカップとハンドルを判定する。成立しなければ None。
    trace=True なら、成立しなかったときに落ちた段階の名前（STAGES のどれか）を返す。
    """
    fail = (lambda stage: stage) if trace else (lambda stage: None)
    p = params
    H, L, C = ind["high"], ind["low"], ind["close"]
    a50L = ind["atr50"][i_left]
    a50t = ind["atr50"][t]
    if not (np.isfinite(a50L) and np.isfinite(a50t)) or a50L <= 0 or a50t <= 0:
        return fail("ATR")
    h_left = H[i_left]

    # 4.2 事前の上昇: i_left 以前60本の最安値からの上昇幅
    lb = p["prior_uptrend_lookback"]
    if i_left < lb:
        return fail("事前上昇")
    l_prior = L[i_left - lb:i_left].min()
    prior_rise = (h_left - l_prior) / a50L
    if prior_rise < p["prior_uptrend_min_atr"]:
        return fail("事前上昇")

    # 4.3 カップの底と右縁
    i_cup = i_left + int(np.argmin(L[i_left:t + 1]))
    l_cup = L[i_cup]
    if i_cup <= i_left:
        return fail("カップの底")
    i_right = i_cup + int(np.argmax(H[i_cup:t + 1]))
    h_right = H[i_right]
    handle_len = t - i_right
    cup_len = i_right - i_left
    if not (p["handle_len_min"] <= handle_len <= p["handle_len_max"]):
        return fail("ハンドル長")
    if not (p["cup_len_min"] <= cup_len <= p["cup_len_max"]):
        return fail("カップ長")
    if p.get("left_rim_highest", True) and H[i_left + 1:i_cup + 1].max() > h_left:
        return fail("左縁が最高値")
    depth_atr = (h_left - l_cup) / a50L
    if not (p["cup_depth_min_atr"] <= depth_atr <= p["cup_depth_max_atr"]):
        return fail("深さ")
    pos = (i_cup - i_left) / cup_len
    if not (p["cup_low_pos_min"] <= pos <= p["cup_low_pos_max"]):
        return fail("底の位置")
    round_ratio = np.count_nonzero(L[i_left:i_right + 1] <= l_cup + p["cup_round_band_atr"] * a50L) / cup_len
    if round_ratio < p["cup_round_min_ratio"]:
        return fail("丸み")
    tol = p["right_rim_tolerance_atr"] * a50L
    if h_right < h_left - tol:
        return fail("右縁(下限)")
    if p.get("right_rim_mode", "min") == "range" and h_right > h_left + tol:
        return fail("右縁(上限)")

    # 4.4 ハンドル（i_right の翌足〜t）
    hs = slice(i_right + 1, t + 1)
    l_handle = L[hs].min()
    pullback_atr = (h_right - l_handle) / a50t
    if not (p["handle_pullback_min_atr"] <= pullback_atr <= p["handle_pullback_max_atr"]):
        return fail("押し幅")
    if pullback_atr > p["handle_pullback_max_ratio"] * depth_atr:
        return fail("押し/深さ")
    if l_handle < l_cup + 0.5 * (h_left - l_cup):
        return fail("ハンドル位置")
    atr14_h = ind["atr14"][hs]
    if not np.all(np.isfinite(atr14_h)) or atr14_h.mean() >= p["handle_atr_contraction"] * a50t:
        return fail("ATR収縮")
    if C[hs].max() > h_right:
        return fail("ハンドル中の終値")

    d = ind["direction"]
    return Setup(
        i_left=i_left, i_cup=i_cup, i_right=i_right, i_handle_end=t,
        h_left=d * h_left, l_cup=d * l_cup, h_right=d * h_right, l_handle=d * l_handle,
        depth_atr=float(depth_atr), pullback_atr=float(pullback_atr), direction=d,
        cup_len=cup_len, handle_len=handle_len, round_ratio=float(round_ratio), prior_rise_atr=float(prior_rise),
    )


def detect_at(ind, params, t):
    """prepare() 済みの配列で、足 t の確定時点のセットアップを返す（無ければ None）"""
    if not trend_ok(ind, params, t):
        return None
    w = params.get("swing_window", 5)
    # 右縁は t−handle_len_max〜t−handle_len_min、カップの長さは cup_len_min〜cup_len_max なので、左縁はこの範囲にある
    lo = max(t - params["handle_len_max"] - params["cup_len_max"], params["prior_uptrend_lookback"])
    hi = min(t - params["handle_len_min"] - params["cup_len_min"], t - w)
    if hi < lo:
        return None
    cands = np.flatnonzero(ind["swing"][lo:hi + 1]) + lo
    for i_left in cands[::-1]:          # 新しい左縁から順に試す
        s = _check_left(ind, params, t, int(i_left))
        if s is not None:
            return s
    return None


def detect_cup_with_handle(df: pd.DataFrame, params: dict, t: int, direction: int = 1):
    """時点tまでのデータのみを使って検出する。df.iloc[:t+1] 以外を参照しない。"""
    ind = prepare(df.iloc[:t + 1], params, direction)
    return detect_at(ind, params, t)
