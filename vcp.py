"""
ミネルヴィニの VCP（Volatility Contraction Pattern）の「押しの縮小」を4時間足で判定する。

定義（買い。売りはすべて逆）
  ベース     : エントリーライン（前後 window 本の Swing High。main_4H_fixedSL.compute_signals と同じ）を付けた足から今まで
  押し T1..Tk: ベースの中で確定した4時間足のスイング（dow_trend と同じ採用ルール。既定は左右2本・ATR(14)×0.5）を順に見て、
               安値ごとに「直前の安値（T1 はベースの起点）からその安値までの最高値 → その安値」を1回の押しとする
  深さ       : 押しの高値 − 安値（価格）。表示用に ATR(EMA, エントリーと同じ) の倍数と % も返す
  VCP 成立   : 押しが min_count〜max_count 回、深さが毎回前より浅い、最後の押しの深さ ≤ ATR × last_max_atr

各4時間足の値は、1本前の4時間足の終値までに確定した情報だけで決まる（エントリーラインと同じ1本ずらし）。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dow_trend import DowConfig, atr_wilder, _update_pivots
from main_4H_fixedSL import get_swing_high_marks, get_swing_low_marks, calculate_atr_ema


@dataclass(frozen=True)
class VcpConfig:
    window: int = 18             # ベースの起点（エントリーライン）の前後本数
    atr_period: int = 18         # 深さを測る ATR（EMA）。エントリーの SL/TP と同じ
    swing: DowConfig = DowConfig(n=2, atr_period=14, min_swing_atr=0.5)  # 押しを取り出すスイング
    min_count: int = 2
    max_count: int = 3
    last_max_atr: float = 1.5    # 最後の押しの深さの上限（ATR の倍数）


def _base_starts(marks, window):
    """各足 t で使えるベースの起点（t - window までに確定した直近のスイングの足番号。無ければ -1）"""
    out = np.full(len(marks), -1)
    last = -1
    for t in range(len(marks)):
        p = t - window
        if p >= 0 and not np.isnan(marks[p]):
            last = p
        out[t] = last
    return out


def _contractions(piv, start, start_price, up):
    """ベースの起点より後のピボットから押し（買い）/ 戻り（売り）を取り出す。[(高値側, 安値側)] のリスト"""
    seq = []
    for p in reversed(piv):
        if p["bar"] <= start:
            break
        seq.append(p)
    seq.reverse()
    top, bottom = ("H", "L") if up else ("L", "H")
    segs, ext = [], dict(bar=start, price=start_price)
    for p in seq:
        if p["kind"] == top:
            if ext is None or (p["price"] > ext["price"] if up else p["price"] < ext["price"]):
                ext = p
        elif ext is not None:
            segs.append((ext, p))
            ext = None
        elif segs and (p["price"] < segs[-1][1]["price"] if up else p["price"] > segs[-1][1]["price"]):
            segs[-1] = (segs[-1][0], p)    # 同じ向きのピボットが続いたら（置き換え後など）、より深い方にする
    return segs


def vcp_scan(h4, cfg: VcpConfig = VcpConfig(), price_decimals=5, queries=None):
    """
    Returns
    -------
    flags : DataFrame（index = h4.index。その足の間に使う値）
        VcpLong / VcpShort（VCP 成立なら 1）、NLong / NShort（押し/戻りの回数）、LastLong / LastShort（最後の深さ ÷ ATR）、
        DecLong / DecShort（深さが毎回前より浅ければ 1）
    details : queries=[(4時間足の開始時刻, "L" or "H")] を渡したときだけ、その足で判定に使った内容のリスト
        dict(kind, base=dict(time, price), segs=[dict(hi_time, hi, lo_time, lo, depth_atr, depth_pct, n)], atr, ok)
    """
    h, l, c = (h4[k].to_numpy(float) for k in ("High", "Low", "Close"))
    atr_swing = atr_wilder(h, l, c, cfg.swing.atr_period)
    atr = calculate_atr_ema(h, l, c, cfg.atr_period, price_decimals)
    start_h = _base_starts(get_swing_high_marks(h, cfg.window), cfg.window)
    start_l = _base_starts(get_swing_low_marks(l, cfg.window), cfg.window)
    want = {}
    for q, (ts, side) in enumerate(queries or ()):
        want.setdefault(h4.index.get_loc(pd.Timestamp(ts)) - 1, []).append((q, side))
    details = [None] * len(queries or ())
    flags = np.full((len(c), 8), np.nan)
    piv = []
    for i in range(len(c)):
        _update_pivots(piv, i, h, l, atr_swing, cfg.swing)
        for col, up, start, prices in ((0, True, start_h[i], h), (1, False, start_l[i], l)):
            if start < 0 or np.isnan(atr[i]):
                continue
            segs = _contractions(piv, start, prices[start], up)
            depth = [abs(a["price"] - b["price"]) for a, b in segs]
            ok = (cfg.min_count <= len(depth) <= cfg.max_count
                  and all(y < x for x, y in zip(depth, depth[1:]))
                  and depth[-1] <= cfg.last_max_atr * atr[i])
            flags[i, col] = float(ok)
            flags[i, col + 2] = len(depth)
            flags[i, col + 4] = depth[-1] / atr[i] if depth else np.nan
            flags[i, col + 6] = float(all(y < x for x, y in zip(depth, depth[1:])))
            for q, side in want.get(i, ()):
                if (side == "L") != up:
                    continue
                details[q] = dict(
                    kind="L" if up else "H", ok=bool(ok), atr=float(atr[i]),
                    base=dict(time=h4.index[start], price=float(prices[start])),
                    segs=[dict(hi_time=h4.index[a["bar"]], hi=float(a["price"]), lo_time=h4.index[b["bar"]],
                               lo=float(b["price"]), depth_atr=float(d / atr[i]),
                               depth_pct=float(d / a["price"] * 100), n=k + 1)
                          for k, ((a, b), d) in enumerate(zip(segs, depth))])
    f = pd.DataFrame(flags, index=h4.index, columns=["VcpLong", "VcpShort", "NLong", "NShort", "LastLong", "LastShort",
                                                       "DecLong", "DecShort"])
    return f.shift(1), details
