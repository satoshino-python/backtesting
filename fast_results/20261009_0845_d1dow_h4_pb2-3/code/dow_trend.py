"""
ダウ理論ベースのトレンド判定（高値・安値の切り上げ/切り下げ）。

仕様の詳細は docs/DOW_TREND_SPEC.md を参照。
入力は Open/High/Low/Close 列を持つ DataFrame（backtesting.py と同じ列名。小文字も可）。
バー番号 i の判定は、バー i までのデータだけで決まる（未来のデータは使わない）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UP, RANGE, DOWN = 1, 0, -1


@dataclass(frozen=True)
class DowConfig:
    n: int = 3                 # ピボット幅（左右の本数）
    atr_period: int = 14       # ATR 期間
    min_swing_atr: float = 1.0  # 最小スイング幅（ATR の倍数）。0 でフィルターなし
    use_wick: bool = True      # True: ヒゲ（高値/安値）でブレイク判定。False: 終値で判定


def _ohlc(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    return tuple(df[cols[k]].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))


def atr_wilder(high, low, close, period: int = 14) -> np.ndarray:
    """ワイルダー平滑化の ATR。最初の period 本は NaN、period 本目で単純平均を初期値にする。"""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = tr[:period].mean()
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _update_pivots(piv: list, i: int, high, low, atr, cfg: DowConfig) -> None:
    """バー i の時点で確定するピボット（バー i - n）を、採用ルールに従って piv に反映する"""
    n = cfg.n
    j = i - n                   # j 番目のバーは、右側 n 本が揃うこの時点で確定する
    if j >= n:
        win_h = high[j - n:j + n + 1]
        win_l = low[j - n:j + n + 1]
        cands = []
        # 左右 n 本の高値(安値)より厳密に高い(低い)ものだけ。同値は不採用
        if high[j] == win_h.max() and (win_h == high[j]).sum() == 1:
            cands.append(("H", j, high[j]))
        if low[j] == win_l.min() and (win_l == low[j]).sum() == 1:
            cands.append(("L", j, low[j]))
        for kind, idx, price in cands:
            c = dict(kind=kind, bar=idx, price=price, confirmed_bar=i)
            if not piv:
                piv.append(c)
                continue
            last = piv[-1]
            if last["kind"] == kind:
                # 同種が連続したら、より極端な方に置き換える
                if (kind == "H" and price > last["price"]) or (kind == "L" and price < last["price"]):
                    piv[-1] = c
            else:
                a = atr[i] if not np.isnan(atr[i]) else 0.0
                if abs(price - last["price"]) >= cfg.min_swing_atr * a:
                    piv.append(c)


def compute_dow_trend(df: pd.DataFrame, cfg: DowConfig = DowConfig()):
    """
    Returns
    -------
    trend : pd.Series[int]
        各バーの判定。1=上昇, 0=レンジ/判定なし, -1=下降。
    pivots : pd.DataFrame
        採用されたスイング。列: kind('H'/'L'), bar(ピボットのバー番号), price,
        confirmed_bar(確定したバー番号。bar + n 以降), time, confirmed_time。
        後から置き換えられたピボットは含まれない（最終的に残ったものだけ）。
    """
    _, high, low, close = _ohlc(df)
    open_ = _ohlc(df)[0]
    n_bars, n = len(close), cfg.n
    atr = atr_wilder(high, low, close, cfg.atr_period)

    piv: list[dict] = []            # 採用済みピボット（H と L が交互に並ぶ）
    trend = np.zeros(n_bars, dtype=int)
    state = RANGE

    for i in range(n_bars):
        _update_pivots(piv, i, high, low, atr, cfg)

        # 確定済みの直近スイングハイ/ロー（piv 内の位置）
        qh = next((q for q in range(len(piv) - 1, -1, -1) if piv[q]["kind"] == "H"), -1)
        ql = next((q for q in range(len(piv) - 1, -1, -1) if piv[q]["kind"] == "L"), -1)
        if qh >= 0 and ql >= 0:
            sh, sl = piv[qh], piv[ql]
            px_lo = low[i] if cfg.use_wick else close[i]
            px_hi = high[i] if cfg.use_wick else close[i]
            broke_dn, broke_up = px_lo < sl["price"], px_hi > sh["price"]
            if broke_dn and broke_up:           # 両方抜けた足は陽線なら上、陰線なら下
                direction = UP if close[i] >= open_[i] else DOWN
            else:
                direction = DOWN if broke_dn else UP if broke_up else 0

            if direction == DOWN:
                # 安値更新。割る前の戻り高値 < その前のスイングハイ なら高値・安値とも切り下げ
                prev = piv[ql - 1] if ql >= 1 else None
                rally_high = high[sl["bar"]:i].max() if i > sl["bar"] else -np.inf
                if prev and prev["kind"] == "H" and rally_high < prev["price"]:
                    state = DOWN
                elif state == UP:
                    state = RANGE
            elif direction == UP:
                prev = piv[qh - 1] if qh >= 1 else None
                pullback_low = low[sh["bar"]:i].min() if i > sh["bar"] else np.inf
                if prev and prev["kind"] == "L" and pullback_low > prev["price"]:
                    state = UP
                elif state == DOWN:
                    state = RANGE
        trend[i] = state

    idx = df.index
    pivots = pd.DataFrame(piv, columns=["kind", "bar", "price", "confirmed_bar"])
    if len(pivots):
        pivots["time"] = idx[pivots["bar"].to_numpy()]
        pivots["confirmed_time"] = idx[pivots["confirmed_bar"].to_numpy()]
    return pd.Series(trend, index=idx, name="dow_trend"), pivots


def swing_structure(df: pd.DataFrame, cfg: DowConfig = DowConfig()) -> pd.DataFrame:
    """
    各バーの終値時点で確定済みのスイング構造（compute_dow_trend と同じピボット採用ルール）。

    列: H1 / H0 = 直近 / 1つ前のスイングハイの価格、L1 / L0 = 直近 / 1つ前のスイングローの価格、
        H1Intact / L1Intact = H1 を超える高値（L1 を下回る安値）が H1（L1）のバーの後、このバーまでに出ていなければ 1。
    まだ無いものは NaN。バー i の値はバー i までのデータだけで決まる（使うときは1本ずらす）。
    """
    _, high, low, close = _ohlc(df)
    n_bars = len(close)
    atr = atr_wilder(high, low, close, cfg.atr_period)
    out = np.full((n_bars, 6), np.nan)
    piv: list[dict] = []
    cur = {"H": None, "L": None}        # 追跡中の直近ピボットのバー番号
    ext = {"H": -np.inf, "L": np.inf}   # そのピボットの後の最高値 / 最安値
    for i in range(n_bars):
        _update_pivots(piv, i, high, low, atr, cfg)
        for kind, arr, agg in (("H", high, np.max), ("L", low, np.min)):
            same = [p for p in piv[-4:] if p["kind"] == kind]
            if not same:
                continue
            last = same[-1]
            if cur[kind] != last["bar"]:
                cur[kind] = last["bar"]
                seg = arr[last["bar"] + 1:i + 1]
                ext[kind] = agg(seg) if len(seg) else (-np.inf if kind == "H" else np.inf)
            else:
                ext[kind] = max(ext[kind], arr[i]) if kind == "H" else min(ext[kind], arr[i])
            col = 0 if kind == "H" else 2
            out[i, col] = last["price"]
            if len(same) >= 2:
                out[i, col + 1] = same[-2]["price"]
            intact = ext[kind] <= last["price"] if kind == "H" else ext[kind] >= last["price"]
            out[i, 4 if kind == "H" else 5] = float(intact)
    return pd.DataFrame(out, index=df.index, columns=["H1", "H0", "L1", "L0", "H1Intact", "L1Intact"])
