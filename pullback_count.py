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


def pullback_waves(h4, d1, d1_cfg: DowConfig, h4_cfg: DowConfig, queries, session_start_hour=17):
    """
    指定した4時間足でカウントに使った波の位置を返す（チャート表示用。pullback_counts と同じ数え方）。

    queries: [(4時間足の開始時刻, "L" or "H")]。"L" = 買い（日足の安値を起点に4Hの押しを数える）、"H" = 売り
    戻り値: queries と同じ順のリスト。各要素は dict
      kind = "L"（買い。押しを数える）/ "H"（売り。戻りを数える）
      anchor = 起点の日足スイング {time（その日の4時間足のうち極値を付けた足の開始時刻）, price, day（日足の日付）}
      points = 数えた4時間足のスイング [{time, price, n}]（n = 1, 2, 3…）
    起点が無い場合は None。
    """
    h, l, c = (h4[k].to_numpy(float) for k in ("High", "Low", "Close"))
    atr = atr_wilder(h, l, c, h4_cfg.atr_period)
    bar_day = get_trading_day_label(h4.index, session_start_hour)
    anc = daily_anchors(d1, d1_cfg).reindex(bar_day)
    want = {}
    for q, (ts, side) in enumerate(queries):
        want.setdefault(h4.index.get_loc(pd.Timestamp(ts)), []).append((q, side))
    out = [None] * len(queries)
    piv = []
    for j in range(len(c)):
        if j >= 1:
            _update_pivots(piv, j - 1, h, l, atr, h4_cfg)
        for q, side in want.get(j, ()):
            kind = "L" if side == "L" else "H"
            day, px = (anc.LDay.iloc[j], anc.LPrice.iloc[j]) if kind == "L" else (anc.HDay.iloc[j], anc.HPrice.iloc[j])
            if pd.isna(day):
                continue
            same_day = np.flatnonzero(bar_day == day)
            ext = (h4.Low if kind == "L" else h4.High).to_numpy(float)[same_day]
            a_bar = same_day[int(np.argmin(np.abs(ext - px)))]
            pts = []
            for p in reversed(piv):
                if bar_day[p["bar"]] <= day:
                    break
                if p["kind"] == kind:
                    pts.append(p)
            pts.reverse()
            out[q] = dict(kind=kind, anchor=dict(time=h4.index[a_bar], price=float(px), day=pd.Timestamp(day)),
                          points=[dict(time=h4.index[p["bar"]], price=float(p["price"]), n=n + 1)
                                  for n, p in enumerate(pts)])
    return out
