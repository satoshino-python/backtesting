"""
スイングブレイクアウト・トレンドフォロー戦略（docs/BREAKOUT_TREND_SPEC.md）の売買シミュレーションと集計。

4時間足（resample_to_signal_bars() の Open/High/Low/Close）だけで判定・約定する。backtesting.py は使わない。
- フラクタル（左右 fractal_n 本より厳密に高い/低い）のスイングハイ/ローを、終値で抜けたら次の足の始値でエントリー
- 初期損切り・時間切れ撤退・建値移動・トレイリング（最高値 − ATR 倍 と 直近スイングの近い方）・最大保有で決済
- 損益は R（= 初期損切り幅）で評価する

使い方:
    trades = simulate(bars, BreakoutParams(), spread=0.00008, symbol="EURUSD",
                      start="2021-01-01", end="2025-12-31")
    summary(trades)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from dow_trend import atr_wilder
from main_4H_fixedSL import signal_bar_trading_day

REASONS = {"initial": "初期損切り", "time": "時間切れ", "breakeven": "建値", "trail": "トレイル",
           "max": "最大保有", "end": "期末"}


@dataclass(frozen=True)
class BreakoutParams:
    fractal_n: int = 3          # フラクタル左右本数
    atr_period: int = 14        # ATR 期間（Wilder）
    sl_atr: float = 1.5         # 初期損切り（ATR 倍）。1R
    time_bars: int = 6          # 時間切れ判定の本数（エントリー足を1本目）
    time_atr: float = 1.0       # 時間切れ到達基準（ATR 倍）
    be_atr: float = 1.5         # 建値移動（ATR 倍）
    trail_atr: float = 2.0      # トレイル幅（最高値 − ATR 倍）
    trail_swing: bool = True    # エントリー後に確定したスイングにも追随する
    max_bars: int = 60          # 最大保有本数
    atr_mode: str = "fixed"     # "fixed": シグナル足の ATR で固定 / "update": 毎バー更新（初期損切りと 1R は固定）
    # 4時間足の移動平均（終値の単純移動平均）の並びによる方向フィルター。(短期, 中期, 長期) の期間。None で無効。
    # 買い: 短期 > 中期 > 長期 のときだけ / 売り: 短期 < 中期 < 長期 のときだけ。判定はシグナル足の確定時点の値
    ma_filter: tuple | None = None

    def label(self):
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


def fractal_pivots(high, low, n):
    """左右 n 本より厳密に高い（低い）足を True にした配列（ピボットの足の位置。確定は n 本後）"""
    m = len(high)
    sh = np.zeros(m, dtype=bool)
    sl = np.zeros(m, dtype=bool)
    if m < 2 * n + 1:
        return sh, sl
    core_h, core_l = high[n:m - n], low[n:m - n]
    ok_h = np.ones(m - 2 * n, dtype=bool)
    ok_l = np.ones(m - 2 * n, dtype=bool)
    for k in range(1, n + 1):
        ok_h &= (core_h > high[n - k:m - n - k]) & (core_h > high[n + k:m - n + k])
        ok_l &= (core_l < low[n - k:m - n - k]) & (core_l < low[n + k:m - n + k])
    sh[n:m - n], sl[n:m - n] = ok_h, ok_l
    return sh, sl


def ma_filter_allow(close, periods):
    """
    移動平均の並びによる方向フィルター。バー t の確定時点（終値まで）の SMA で判定する。
    Returns: (買いを許可, 売りを許可) の bool 配列。SMA が計算できない足（長期の本数に満たない）は両方 False。
    """
    c = pd.Series(np.asarray(close, dtype=float))
    fast, mid, slow = (c.rolling(k).mean().to_numpy() for k in periods)
    return (fast > mid) & (mid > slow), (fast < mid) & (mid < slow)   # NaN との比較は False


def simulate(bars, p: BreakoutParams = BreakoutParams(), *, spread=0.0, slippage=0.0, cost_mult=1.0,
             symbol="", start=None, end=None, session_start_hour=17):
    """
    bars  : 4時間足（index = 足の開始時刻、列 Open/High/Low/Close）。助走期間を含めて渡す
    spread / slippage : 価格単位。1回の約定ごとに spread/2 + slippage を不利な方向に乗せる（cost_mult 倍）
    start / end : エントリーできるシグナル足の取引日の範囲（"YYYY-MM-DD"）。end の最後の足で保有中なら終値で決済
    Returns: トレード一覧の DataFrame（1行 = 1トレード）
    """
    if p.atr_mode not in ("fixed", "update"):
        raise ValueError(f"atr_mode は 'fixed' / 'update': {p.atr_mode}")
    day = signal_bar_trading_day(bars.index, session_start_hour)
    in_range = np.ones(len(bars), dtype=bool)
    if start is not None:
        in_range &= day >= pd.Timestamp(start)
    if end is not None:
        in_range &= day <= pd.Timestamp(end)
    last = int(np.flatnonzero(in_range)[-1]) if in_range.any() else -1
    o, h, l, c = (bars[k].to_numpy(dtype=float) for k in ("Open", "High", "Low", "Close"))
    atr = atr_wilder(h, l, c, p.atr_period)
    is_sh, is_sl = fractal_pivots(h, l, p.fractal_n)
    cost = (spread / 2 + slippage) * cost_mult
    if p.ma_filter is not None:
        allow = dict(zip((1, -1), ma_filter_allow(c, p.ma_filter)))
    else:
        allow = {1: np.ones(len(bars), bool), -1: np.ones(len(bars), bool)}
    n = p.fractal_n
    times = bars.index

    trades = []
    pos = None              # 保有中のポジション（dict）
    pending_entry = None    # (方向, シグナル足, ATR)
    pending_exit = None     # 決済理由
    ref = {1: None, -1: None}   # 基準価格（直近確定スイング）: [価格, 使用済み]
    last_swing = {1: None, -1: None}  # 直近に確定したスイング（トレイル用）: (価格, 確定足)

    def close_pos(t, raw_price, reason, fav_ext, adv_ext):
        nonlocal pos
        d = pos["d"]
        exit_price = raw_price - d * cost
        # 有利側は d を掛けて大きい方、不利側は小さい方（売りでは上下が逆になる）
        best = max(pos["best"] * d, fav_ext * d) * d if fav_ext is not None else pos["best"]
        worst = min(pos["worst"] * d, adv_ext * d) * d if adv_ext is not None else pos["worst"]
        a = pos["atr"]
        pnl = (exit_price - pos["entry"]) * d
        trades.append(dict(
            Symbol=symbol, Side="買い" if d == 1 else "売り", Dir=d,
            SignalTime=times[pos["sig"]], EntryTime=times[pos["bar"]], EntryPrice=pos["entry"],
            ExitTime=times[t], ExitPrice=exit_price, Reason=REASONS[reason], ReasonCode=reason,
            Bars=t - pos["bar"] + (0 if reason in ("time", "max") else 1),
            ATR=a, InitialStop=pos["init_stop"], FinalStop=pos["stop"],
            PnL=pnl, R=pnl / (p.sl_atr * a),
            MFE_ATR=(best - pos["entry"]) * d / a, MAE_ATR=max(0.0, (pos["entry"] - worst) * d / a),
            BE=pos["be"],
        ))
        pos = None

    for t in range(len(bars)):
        if t > last:
            break
        exited = False
        # ① 予約した決済（時間切れ・最大保有）を始値で
        if pos is not None and pending_exit is not None:
            close_pos(t, o[t], pending_exit, o[t], o[t])
            pending_exit, exited = None, True
        # ② 予約したエントリーを始値で
        if pending_entry is not None:
            d, sig, a = pending_entry
            pending_entry = None
            entry = o[t] + d * cost
            stop = entry - d * p.sl_atr * a
            pos = dict(d=d, sig=sig, bar=t, entry=entry, atr=a, stop=stop, init_stop=stop,
                       best=entry, worst=entry, be=False)
        # ③ ストップ判定（エントリー足も対象。有効なストップは前の足の確定時までに決めたもの）
        if pos is not None:
            d, stop = pos["d"], pos["stop"]
            if (o[t] - stop) * d <= 0:          # 窓開けでストップを越えて始まった
                fill = o[t]
            elif ((l[t] if d == 1 else h[t]) - stop) * d <= 0:
                fill = stop
            else:
                fill = None
            if fill is not None:
                if not pos["be"]:
                    reason = "initial"
                elif abs(stop - pos["entry"]) <= 1e-12 * max(1.0, abs(stop)):
                    reason = "breakeven"
                else:
                    reason = "trail"
                # ストップが先とみなすので、この足の有利側は含めない。不利側は約定価格まで
                close_pos(t, fill, reason, None, fill)
                exited = True
        # スイングの確定（バー t の確定時点で、ピボット t-n が既知になる）
        j = t - n
        if j >= 0:
            if is_sh[j]:
                ref[1] = [h[j], False]
                last_swing[1] = (h[j], t)
            if is_sl[j]:
                ref[-1] = [l[j], False]
                last_swing[-1] = (l[j], t)
        # ④ 足の確定後: 最高値・建値移動・トレイル・時間切れ・最大保有
        if pos is not None:
            d = pos["d"]
            fav, adv = (h[t], l[t]) if d == 1 else (l[t], h[t])
            pos["best"] = fav if (fav - pos["best"]) * d > 0 else pos["best"]
            pos["worst"] = adv if (adv - pos["worst"]) * d < 0 else pos["worst"]
            a_now = pos["atr"] if p.atr_mode == "fixed" or np.isnan(atr[t]) else atr[t]
            gain = (pos["best"] - pos["entry"]) * d
            if not pos["be"] and gain >= p.be_atr * a_now:
                pos["be"] = True
            if pos["be"]:
                cands = [pos["stop"], pos["entry"], pos["best"] - d * p.trail_atr * a_now]
                sw = last_swing[-d]
                if p.trail_swing and sw is not None and sw[1] >= pos["bar"]:
                    cands.append(sw[0])
                pos["stop"] = max(x * d for x in cands) * d
            held = t - pos["bar"] + 1
            if held == p.time_bars and gain < p.time_atr * a_now:
                pending_exit = "time"
            elif held >= p.max_bars:
                pending_exit = "max"
            if t == last:       # 検証期間の最後の足: 終値で決済
                close_pos(t, c[t], "end", c[t], c[t])
                pending_exit, exited = None, True
        # ⑤ シグナル判定（バー t の確定時点）
        if t == 0:
            continue
        for d in (1, -1):
            r = ref[d]
            if r is None or r[1]:
                continue
            if (c[t - 1] - r[0]) * d <= 0 < (c[t] - r[0]) * d:
                r[1] = True     # 一度ブレイクしたスイングは使用済み（エントリーしなくても）
                if (pos is None and not exited and pending_exit is None and in_range[t] and t < last and allow[d][t]
                        and not np.isnan(atr[t])):
                    pending_entry = (d, t, atr[t])

    cols = ["Symbol", "Side", "Dir", "SignalTime", "EntryTime", "EntryPrice", "ExitTime", "ExitPrice",
            "Reason", "ReasonCode", "Bars", "ATR", "InitialStop", "FinalStop", "PnL", "R",
            "MFE_ATR", "MAE_ATR", "BE"]
    return pd.DataFrame(trades, columns=cols)


# ===== 集計 =====

def max_losing_streak(r):
    best = cur = 0
    for x in r:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    return best


def max_drawdown(curve):
    """累積値の系列の最大ドローダウン（ピークからの下落幅。正の値）"""
    curve = np.asarray(curve, dtype=float)
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    return float((peak - curve).max())


def summary(trades):
    """トレード一覧（決済順に並べる）から成績を計算する"""
    t = trades.sort_values("ExitTime") if len(trades) else trades
    r = t["R"].to_numpy(dtype=float) if len(t) else np.array([])
    wins, losses = r[r > 0], r[r <= 0]
    gp, gl = wins.sum(), -losses.sum()
    avg_w = wins.mean() if wins.size else np.nan
    avg_l = losses.mean() if losses.size else np.nan
    return dict(
        trades=int(r.size),
        win_rate=float(wins.size / r.size * 100) if r.size else np.nan,
        avg_win_R=float(avg_w), avg_loss_R=float(avg_l),
        payoff=float(avg_w / -avg_l) if wins.size and losses.size and avg_l < 0 else np.nan,
        expectancy_R=float(r.mean()) if r.size else np.nan,
        total_R=float(r.sum()),
        pf=float(gp / gl) if gl > 0 else np.nan,
        max_dd_R=max_drawdown(np.cumsum(r)),
        max_losing_streak=max_losing_streak(r),
    )


def equity_curve(trades, risk_pct=0.01, initial=1.0):
    """
    固定％リスクの資金曲線（複利）。エントリー時点の確定済み資金 × risk_pct をリスク額とし、
    決済時に R × リスク額を加える。複数銘柄を渡すと1つの口座で同時に持つ扱い。
    Returns: index=決済時刻 の資金の Series（初期値 initial）
    """
    if len(trades) == 0:
        return pd.Series(dtype=float)
    k = np.arange(len(trades))
    entry_t = pd.DatetimeIndex(trades["EntryTime"])
    exit_t = pd.DatetimeIndex(trades["ExitTime"])
    # 同じ時刻では「他のトレードの決済 → エントリー → エントリー足のうちに決済したトレード」の順に処理する
    # （ストップで決済した足は、決済時刻を足の開始時刻で記録しているため）
    ev = pd.concat([
        pd.DataFrame({"time": entry_t, "kind": 1, "k": k}),
        pd.DataFrame({"time": exit_t, "kind": np.where(exit_t == entry_t, 2, 0), "k": k}),
    ]).sort_values(["time", "kind", "k"], kind="stable")
    r = trades["R"].to_numpy(dtype=float)
    eq, risk = initial, np.zeros(len(trades))
    times, values = [], []
    for time, kind, j in ev.itertuples(index=False):
        if kind == 1:
            risk[j] = eq * risk_pct
        else:
            eq += r[j] * risk[j]
            times.append(time)
            values.append(eq)
    return pd.Series(values, index=pd.DatetimeIndex(times), name="equity")


def equity_max_dd_pct(eq, initial=1.0):
    if len(eq) == 0:
        return 0.0
    v = np.concatenate([[initial], eq.to_numpy()])
    peak = np.maximum.accumulate(v)
    return float(((peak - v) / peak).max() * 100)
