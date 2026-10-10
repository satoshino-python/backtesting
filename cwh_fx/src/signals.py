"""
セットアップ → エントリーシグナル（仕様 4.5・5）。

各足 t の確定時に、1本前（t−1）で成立していたセットアップに対して
  close_t > pivot + breakout_buffer_atr × ATR14_t   （ショートは close_t < pivot − 0.1×ATR14_t）
ならシグナル。約定は t+1 の始値（backtest.py で処理）。

セットアップの管理（同じ形 = 同じ左縁・右縁 を key で見分ける）:
  - 初めて成立した足を「ハンドル完成」とし、その足から signal_expiry_bars 本以内にブレイクしなければ失効。
  - 成立後に終値がハンドル安値（成立時点〜前の足までの最安値）を下回ったら破棄。
  - ハンドルが handle_len_max 本を超えたら、detect が成立を返さなくなるので自然に消える。
  - ブレイク・失効・破棄した形は同じ key で再び使わない。
"""
from dataclasses import dataclass

import numpy as np

from .pattern import Setup, detect_at, prepare


@dataclass
class Signal:
    t: int                  # シグナルの足（ブレイクの確定足）
    direction: int
    setup: Setup
    pivot: float
    stop: float             # 初期損切り（L_handle − 0.5×ATR14_t。ショートは H_handle + 0.5×ATR14_t）
    atr14: float            # ATR14_t（追いかけ・損切り幅の判定に使う）
    depth_atr: float
    sma_dist_atr: float     # SMA200 からの乖離（ATR50_t 換算、取引方向を正）。同日に複数シグナルが出たときの優先順位に使う


class SetupTracker:
    def __init__(self, params):
        self.p = params
        self.armed = {}     # key -> {"first": 成立した最初の足, "l_handle": 反転系列でのハンドル安値}
        self.dead = set()
        self.live = None    # 直前の足で成立していたセットアップ

    def step(self, ind, t):
        """足 t の確定時に呼ぶ。シグナルが出たら Signal を返す"""
        p, d = self.p, ind["direction"]
        C = ind["close"]
        sig = None
        prev = self.live
        if prev is not None and prev.key not in self.dead:
            a14 = ind["atr14"][t]
            piv = d * prev.h_right          # 反転系列でのピボット
            if np.isfinite(a14) and C[t] > piv + p["breakout_buffer_atr"] * a14:
                stop_x = d * prev.l_handle - p["stop_buffer_atr"] * a14
                sig = Signal(
                    t=t, direction=d, setup=prev, pivot=prev.h_right, stop=d * stop_x, atr14=float(a14),
                    depth_atr=prev.depth_atr,
                    sma_dist_atr=float((C[t] - ind["sma"][t]) / ind["atr50"][t]),
                )
                self.dead.add(prev.key)
        # 破棄・失効
        for key, st in list(self.armed.items()):
            if key in self.dead:
                del self.armed[key]
            elif C[t] < st["l_handle"] or t - st["first"] >= p["signal_expiry_bars"]:
                self.dead.add(key)
                del self.armed[key]
        # 足 t で成立しているセットアップ
        s = detect_at(ind, p, t)
        if s is not None and s.key not in self.dead:
            lh = d * s.l_handle
            st = self.armed.setdefault(s.key, {"first": t, "l_handle": lh})
            st["l_handle"] = min(st["l_handle"], lh)
            self.live = s
        else:
            self.live = None
        return sig


def generate_signals(df, params, directions=(1, -1)):
    """1つの通貨ペアの全シグナルを返す（ポートフォリオの制約は backtest.py で掛ける）"""
    out = []
    for d in directions:
        ind = prepare(df, params, d)
        tr = SetupTracker(params)
        for t in range(len(df)):
            s = tr.step(ind, t)
            if s is not None:
                out.append(s)
    out.sort(key=lambda s: (s.t, -s.direction))
    return out
