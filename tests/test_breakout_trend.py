"""python tests/test_breakout_trend.py または pytest tests/ で実行。手で作った4時間足で売買ルールを確かめる。"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from breakout_trend import BreakoutParams, fractal_pivots, simulate, equity_curve, ma_filter_allow  # noqa: E402

# ATR が早く使えるように期間を短くする。時間切れ・建値は使わない設定を基本にし、テストごとに変える
P = BreakoutParams(atr_period=3, time_bars=999, be_atr=99.0, max_bars=999)


def make_bars(rows):
    """rows: (open, high, low, close) のリスト"""
    idx = pd.date_range("2024-01-01 17:00", periods=len(rows), freq="4h")
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx, dtype=float)


def flat(n, price=100.0, rng=0.5):
    return [(price, price + rng, price - rng, price)] * n


def base_with_swing_high():
    """
    0-5: 横ばい（高値 100.5）。6: 高値 103 のスイングハイ。7-9: 高値 101 以下（6 は 9 の確定で既知）。
    10: 終値 103.5 で上抜け → 11 の始値でエントリー
    """
    rows = flat(6)
    rows += [(100, 103, 99.5, 100.5)]                    # 6 スイングハイ 103
    rows += [(100.5, 101, 99.5, 100)] * 3                # 7-9
    rows += [(100, 104, 99.8, 103.5)]                    # 10 シグナル足（前の終値 100 ≦ 103 < 103.5）
    return rows


def atr_at(bars, i, period=3):
    from dow_trend import atr_wilder
    return atr_wilder(bars.High.values, bars.Low.values, bars.Close.values, period)[i]


def test_fractal_strict():
    h = np.array([1, 2, 3, 2, 1, 3, 3, 2, 1], dtype=float)
    sh, _ = fractal_pivots(h, h, 2)
    assert sh.tolist() == [False, False, True, False, False, False, False, False, False]  # 同値の 5/6 は不採用


def test_entry_next_open_and_initial_stop():
    rows = base_with_swing_high()
    rows += [(103.6, 104, 103, 103.8)]                   # 11 エントリー（始値 103.6）
    rows += [(103.8, 104, 90, 91)]                       # 12 初期損切り
    rows += flat(3, 91)
    bars = make_bars(rows)
    tr = simulate(bars, P)
    assert len(tr) == 1
    t = tr.iloc[0]
    a = atr_at(bars, 10)
    assert t.Dir == 1 and t.EntryTime == bars.index[11] and t.EntryPrice == 103.6
    assert np.isclose(t.ATR, a)
    assert np.isclose(t.ExitPrice, 103.6 - 1.5 * a) and t.ReasonCode == "initial"
    assert np.isclose(t.R, -1.0) and t.Bars == 2


def test_gap_through_stop_fills_at_open():
    rows = base_with_swing_high()
    rows += [(103.6, 104, 103, 103.8)]
    rows += [(80, 81, 79, 80)]                           # 窓開けでストップより下から始まる
    rows += flat(3, 80)
    tr = simulate(make_bars(rows), P)
    assert tr.iloc[0].ExitPrice == 80 and tr.iloc[0].R < -1


def test_costs_applied_both_sides():
    rows = base_with_swing_high()
    rows += [(103.6, 104, 103, 103.8)]
    rows += [(80, 81, 79, 80)]
    rows += flat(3, 80)
    tr = simulate(make_bars(rows), P, spread=0.2, slippage=0.05)
    assert np.isclose(tr.iloc[0].EntryPrice, 103.6 + 0.15)
    assert np.isclose(tr.iloc[0].ExitPrice, 80 - 0.15)


def test_breakeven_and_stop_same_bar_is_initial_stop():
    """同じ足で +1.5ATR に届き、初期ストップにも届いたら、初期損切り（ストップが先）"""
    rows = base_with_swing_high()
    rows += [(103.6, 140, 80, 100)]                      # 11 エントリー足で両方に届く
    rows += flat(3, 100)
    tr = simulate(make_bars(rows), replace(P, be_atr=1.5))
    assert tr.iloc[0].ReasonCode == "initial" and np.isclose(tr.iloc[0].R, -1.0)
    assert tr.iloc[0].MFE_ATR == 0                       # ストップの足の有利側は含めない


def test_breakeven_moves_next_bar():
    rows = base_with_swing_high()
    bars0 = make_bars(rows)
    a = atr_at(bars0, 10)
    rows += [(103.6, 103.6 + 1.6 * a, 103.5, 104)]       # 11 +1.5ATR 到達（ストップ到達なし）→ 次の足から建値
    rows += [(104, 104.2, 103.0, 103.2)]                 # 12 建値 103.6 に到達
    rows += flat(3, 103.2)
    tr = simulate(make_bars(rows), replace(P, be_atr=1.5, trail_swing=False, trail_atr=99.0))
    t = tr.iloc[0]
    assert t.ReasonCode == "breakeven" and t.ExitPrice == 103.6 and t.R == 0


def test_trailing_from_highest_high():
    rows = base_with_swing_high()
    bars0 = make_bars(rows)
    a = atr_at(bars0, 10)
    hi = 103.6 + 4 * a
    rows += [(103.6, hi, 103.5, hi - 0.1)]               # 11 最高値 → ストップ = hi − 2ATR
    rows += [(hi - 0.1, hi - 0.05, hi - 3 * a, hi - 3 * a)]  # 12 hi − 2ATR を割る
    rows += flat(3, hi - 3 * a)
    tr = simulate(make_bars(rows), replace(P, be_atr=1.5, trail_swing=False))
    t = tr.iloc[0]
    assert t.ReasonCode == "trail" and np.isclose(t.ExitPrice, hi - 2 * a)
    assert np.isclose(t.R, 2 / 1.5 * (4 - 2) / 2)        # (+2ATR) / 1.5ATR


def test_time_exit_next_open():
    rows = base_with_swing_high()
    rows += [(103.6, 103.8, 103.4, 103.6)] * 6           # 11-16: 6本とも +1ATR に届かない
    rows += [(103.7, 103.9, 103.5, 103.6)]               # 17 の始値で決済
    rows += flat(3, 103.6)
    tr = simulate(make_bars(rows), replace(P, time_bars=6))
    t = tr.iloc[0]
    assert t.ReasonCode == "time" and t.ExitTime == make_bars(rows).index[17] and t.ExitPrice == 103.7
    assert t.Bars == 6


def test_max_hold():
    rows = base_with_swing_high()
    rows += [(103.6, 103.8, 103.4, 103.6)] * 10
    tr = simulate(make_bars(rows), replace(P, max_bars=4))
    t = tr.iloc[0]
    assert t.ReasonCode == "max" and t.EntryTime == make_bars(rows).index[11]
    assert t.ExitTime == make_bars(rows).index[15] and t.Bars == 4


def test_swing_not_reused_and_no_entry_on_exit_bar():
    """保有中に抜けたスイングは使用済み。決済した足のシグナルではエントリーしない"""
    rows = base_with_swing_high()
    rows += [(103.6, 104, 103, 103.8)]                   # 11 エントリー
    rows += [(103.8, 104, 90, 91)]                       # 12 損切り
    rows += [(91, 103, 90.5, 102)] + [(102, 102.5, 101, 102)] * 3
    rows += [(102, 104, 101.5, 103.6)]                   # 103 を再び上抜けても、使用済みなのでエントリーしない
    rows += flat(3, 103.6)
    tr = simulate(make_bars(rows), P)
    assert len(tr) == 1


def test_short_is_mirror_of_long():
    rows = base_with_swing_high()
    rows += [(103.6, 104, 103, 103.8), (103.8, 106, 101, 105), (105, 109, 104.5, 108)]
    rows += [(108, 108.5, 95, 96)] + flat(3, 96)
    long_bars = make_bars(rows)
    short_bars = pd.DataFrame({"Open": 200 - long_bars.Open, "High": 200 - long_bars.Low,
                               "Low": 200 - long_bars.High, "Close": 200 - long_bars.Close})
    q = replace(P, be_atr=1.5)
    a, b = simulate(long_bars, q), simulate(short_bars, q)
    assert len(a) == len(b) == 1
    assert (a.Dir.values == -b.Dir.values).all()
    for col in ("R", "MFE_ATR", "MAE_ATR", "Bars"):
        assert np.allclose(a[col].values, b[col].values), col
    assert a.ReasonCode.tolist() == b.ReasonCode.tolist()


def test_ma_filter_allow_arrays():
    up = np.arange(1, 30, dtype=float)
    lo, sh = ma_filter_allow(up, (2, 3, 5))
    assert lo[4:].all() and not sh.any() and not lo[:4].any()      # 長期の本数に満たない足は両方 False
    lo, sh = ma_filter_allow(up[::-1].copy(), (2, 3, 5))
    assert sh[4:].all() and not lo.any()


def test_ma_filter_blocks_and_passes():
    """上昇の並び（短期 > 中期 > 長期）なら買いが通り、並びが逆（期間を逆順に指定）なら見送る"""
    rows = base_with_swing_high() + [(103.6, 104, 103, 103.8), (103.8, 104, 90, 91)] + flat(3, 91)
    bars = make_bars(rows)
    assert len(simulate(bars, replace(P, ma_filter=(2, 3, 5)))) == 1
    assert len(simulate(bars, replace(P, ma_filter=(5, 3, 2)))) == 0


def test_ma_filter_uses_signal_bar_close_only():
    """フィルターはシグナル足の確定時点の SMA で決まる。末尾を切っても、残した範囲のトレードは変わらない"""
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 2500))
    o = np.concatenate([[100], c[:-1]])
    h = np.maximum(o, c) + rng.exponential(0.3, 2500)
    l = np.minimum(o, c) - rng.exponential(0.3, 2500)
    bars = make_bars(list(zip(o, h, l, c)))
    q = BreakoutParams(ma_filter=(20, 75, 200))
    full, none = simulate(bars, q), simulate(bars, BreakoutParams())
    assert 0 < len(full) < len(none)
    part = simulate(bars.iloc[:1800], q)
    done = full[full.ExitTime < bars.index[1799]]
    pd.testing.assert_frame_equal(part.iloc[:len(done)].reset_index(drop=True), done.reset_index(drop=True))
    # 買いは 短期 > 中期 > 長期 のときだけ、売りは逆のときだけ
    sma = lambda k: pd.Series(bars.Close.values).rolling(k).mean().to_numpy()
    pos = {t: i for i, t in enumerate(bars.index)}
    for _, tr in full.iterrows():
        i = pos[tr.SignalTime]
        ok = sma(20)[i] > sma(75)[i] > sma(200)[i] if tr.Dir == 1 else sma(20)[i] < sma(75)[i] < sma(200)[i]
        assert ok


def test_no_lookahead_random_walk():
    """末尾を切ったデータでも、切った位置より前に決済したトレードは全期間と一致する"""
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.5, 3000))
    o = np.concatenate([[100], c[:-1]]) + rng.normal(0, 0.05, 3000)
    h = np.maximum(o, c) + rng.exponential(0.3, 3000)
    l = np.minimum(o, c) - rng.exponential(0.3, 3000)
    bars = make_bars(list(zip(o, h, l, c)))
    full = simulate(bars, BreakoutParams())
    assert len(full) > 50
    for cut in (1000, 1777, 2500):
        part = simulate(bars.iloc[:cut], BreakoutParams())
        done = full[full.ExitTime < bars.index[cut - 1]]
        pd.testing.assert_frame_equal(part.iloc[:len(done)].reset_index(drop=True), done.reset_index(drop=True))


def test_equity_curve_compounds():
    t = pd.DataFrame({"EntryTime": pd.to_datetime(["2024-01-01", "2024-01-03"]),
                      "ExitTime": pd.to_datetime(["2024-01-02", "2024-01-04"]), "R": [2.0, -1.0]})
    eq = equity_curve(t, risk_pct=0.01)
    assert np.allclose(eq.values, [1.02, 1.02 - 0.0102])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
