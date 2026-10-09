"""
「日足の上げ波（下げ波）の中で、4時間足が何回押したか（戻したか）」を4時間足ごとに数える。

定義（買い。売りはすべて逆）
  - 起点: 前日の終値時点で確定している日足の直近スイングロー（dow_trend と同じ採用ルール）。その日を D とする
  - 押し: D より後の取引日にある4時間足のスイングロー（dow_trend と同じ採用ルール）で、1本前の4時間足の終値までに確定したもの
  - PBLong = 押しの数、StairLong = それらの安値が起点から順に切り上がっていれば 1
  売りは 日足の直近スイングハイ を起点に、4時間足のスイングハイ（戻り）を数える（PBShort / StairShort）。

各4時間足の値は、その足の間に使ってよい情報（1本前の4時間足の終値・前日の日足の終値まで）だけで決まる。
"""
import numpy as np
import pandas as pd

from dow_trend import DowConfig, atr_wilder, _update_pivots
from main_4H_fixedSL import get_trading_day_label


def daily_anchors(d1, cfg: DowConfig):
    """各取引日に使う起点（前日の終値で確定した日足の直近スイングロー/ハイの日付と価格）。index = d1.index"""
    h, l, c = (d1[k].to_numpy(float) for k in ("High", "Low", "Close"))
    atr = atr_wilder(h, l, c, cfg.atr_period)
    rows, piv = [], []
    for i in range(len(c)):
        _update_pivots(piv, i, h, l, atr, cfg)
        lo = next((p for p in reversed(piv) if p["kind"] == "L"), None)
        hi = next((p for p in reversed(piv) if p["kind"] == "H"), None)
        rows.append((d1.index[lo["bar"]] if lo else pd.NaT, lo["price"] if lo else np.nan,
                     d1.index[hi["bar"]] if hi else pd.NaT, hi["price"] if hi else np.nan))
    a = pd.DataFrame(rows, index=d1.index, columns=["LDay", "LPrice", "HDay", "HPrice"])
    return a.shift(1)                      # 前日の終値で確定した値をその日に使う


def pullback_counts(h4, d1, d1_cfg: DowConfig, h4_cfg: DowConfig, session_start_hour=17):
    """4時間足ごとの押し/戻りの回数（PBLong / PBShort）と切り上げ/切り下げの判定（StairLong / StairShort）"""
    h, l, c = (h4[k].to_numpy(float) for k in ("High", "Low", "Close"))
    atr = atr_wilder(h, l, c, h4_cfg.atr_period)
    bar_day = get_trading_day_label(h4.index, session_start_hour)
    anc = daily_anchors(d1, d1_cfg).reindex(bar_day)
    l_day, l_px = anc.LDay.to_numpy(), anc.LPrice.to_numpy(float)
    h_day, h_px = anc.HDay.to_numpy(), anc.HPrice.to_numpy(float)
    out = np.full((len(c), 4), np.nan)
    piv = []
    for j in range(len(c)):
        # 4時間足 j の値は、j-1 までに確定したピボットで数える
        if j >= 1:
            _update_pivots(piv, j - 1, h, l, atr, h4_cfg)
        for col, kind, day, px, up in ((0, "L", l_day[j], l_px[j], True), (1, "H", h_day[j], h_px[j], False)):
            if pd.isna(day):
                continue
            prices = []
            for p in reversed(piv):            # ピボットはバー順に並んでいるので、起点の日まで遡れば十分
                if bar_day[p["bar"]] <= day:
                    break
                if p["kind"] == kind:
                    prices.append(p["price"])
            prices.reverse()
            out[j, col] = len(prices)
            seq = [px] + prices
            out[j, col + 2] = float(all((b > a) if up else (b < a) for a, b in zip(seq, seq[1:])))
    return pd.DataFrame(out, index=h4.index, columns=["PBLong", "PBShort", "StairLong", "StairShort"])
