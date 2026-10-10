"""
シグナルが少ないときに「どの条件で落ちているか」を調べる。
  - funnel(): 足ごとに、いちばん先まで進んだ左縁の候補が落ちた段階を数える
  - leave_one_out(): 条件を1つだけ外したときのシグナル数
"""
from collections import Counter

import numpy as np

from .pattern import STAGES, _check_left, prepare, trend_ok
from .signals import generate_signals

ORDER = ("トレンド", "左縁の候補なし") + STAGES + ("成立",)


def funnel(df, params, direction=1, t_from=0):
    ind = prepare(df, params, direction)
    w = params.get("swing_window", 5)
    out = Counter()
    for t in range(t_from, len(df)):
        if not trend_ok(ind, params, t):
            out["トレンド"] += 1
            continue
        lo = max(t - params["handle_len_max"] - params["cup_len_max"], params["prior_uptrend_lookback"])
        hi = min(t - params["handle_len_min"] - params["cup_len_min"], t - w)
        cands = np.flatnonzero(ind["swing"][lo:hi + 1]) + lo if hi >= lo else []
        if len(cands) == 0:
            out["左縁の候補なし"] += 1
            continue
        best = -1
        for i in cands:
            r = _check_left(ind, params, t, int(i), trace=True)
            k = len(STAGES) if not isinstance(r, str) else STAGES.index(r)
            best = max(best, k)
        out["成立" if best == len(STAGES) else STAGES[best]] += 1
    return out


# 条件を1つ外す = その条件が必ず通る値にする
RELAX = {
    "トレンド（SMA200）": {"use_trend_filter": False},
    "事前上昇 ≥6ATR": {"prior_uptrend_min_atr": -1e9},
    "カップ長 30〜150": {"cup_len_min": 6, "cup_len_max": 400},
    "左縁が最高値": {"left_rim_highest": False},
    "深さ 3〜10ATR": {"cup_depth_min_atr": 0.0, "cup_depth_max_atr": 1e9},
    "底の位置 0.25〜0.75": {"cup_low_pos_min": 0.0, "cup_low_pos_max": 1.0},
    "丸み ≥0.20": {"cup_round_min_ratio": 0.0},
    "右縁 −1ATR": {"right_rim_tolerance_atr": 1e9},
    "押し幅 0.5〜2.5ATR": {"handle_pullback_min_atr": 0.0, "handle_pullback_max_atr": 1e9},
    "押し ≤ 深さ×0.5": {"handle_pullback_max_ratio": 1e9},
    "ATR収縮 <0.8": {"handle_atr_contraction": 1e9},
    "ハンドル長 5〜25": {"handle_len_min": 1, "handle_len_max": 60},
    "失効 10本": {"signal_expiry_bars": 10_000},
}


def count_signals(bars_by_pair, params, start=None, end=None):
    n = 0
    for df in bars_by_pair.values():
        sigs = generate_signals(df, params, tuple(params.get("directions", (1, -1))))
        n += sum(1 for s in sigs if (start is None or df.index[s.t] >= start) and (end is None or df.index[s.t] <= end))
    return n


def _loo_one(args):
    name, over, bars_by_pair, params, start, end = args
    return name, count_signals(bars_by_pair, dict(params, **over), start, end)


def leave_one_out(bars_by_pair, params, start=None, end=None, workers=4):
    from concurrent.futures import ProcessPoolExecutor
    items = [("（すべての条件）", {}, bars_by_pair, params, start, end)]
    items += [(k, v, bars_by_pair, params, start, end) for k, v in RELAX.items()]
    with ProcessPoolExecutor(workers) as ex:
        return dict(ex.map(_loo_one, items))
