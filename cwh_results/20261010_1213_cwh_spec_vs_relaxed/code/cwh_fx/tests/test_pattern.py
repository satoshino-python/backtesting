"""合成データでパターン検出を検証する（仕様 12.1）。python cwh_fx/tests/test_pattern.py または pytest で実行"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cwh_fx.src.pattern import detect_at, detect_cup_with_handle, prepare
from cwh_fx.src.signals import generate_signals
from cwh_fx.tests.synthetic import load_params, make_cup

P = load_params()
IDEAL = dict(depth=0.06, range_=0.006)


def _detect(df, info, params=P):
    return detect_at(prepare(df, params), params, info["handle_end"])


def test_ideal_cup_is_detected():
    df, info = make_cup("u", **IDEAL)
    s = _detect(df, info)
    assert s is not None
    assert s.i_left in (info["i_left"], info["i_left"] + 1)   # 左縁の高値が2本で同値になる（新しい方を採用）
    assert 3 <= s.depth_atr <= 10 and 0.5 <= s.pullback_atr <= 2.5
    assert s.handle_len == 10 and 30 <= s.cup_len <= 150
    assert s.h_right <= s.h_left and s.l_handle > s.l_cup


def test_ideal_cup_breakout_signal_and_stop():
    df, info = make_cup("u", **IDEAL)
    sigs = generate_signals(df, P, directions=(1,))
    assert [s.t for s in sigs] == [info["breakout"]]
    s = sigs[0]
    assert df["close"].iloc[s.t] > s.pivot + P["breakout_buffer_atr"] * s.atr14
    assert np.isclose(s.stop, s.setup.l_handle - P["stop_buffer_atr"] * s.atr14)


def test_v_shape_is_rejected_by_roundness():
    df, info = make_cup("v", **IDEAL)
    assert _detect(df, info) is None
    assert _detect(df, info, dict(P, cup_round_min_ratio=0.0)) is not None   # 丸みの条件だけで落ちている


def test_handle_too_long_or_too_short():
    df, info = make_cup("u", handle_len=30, **IDEAL)
    assert _detect(df, info) is None
    assert generate_signals(df, P, directions=(1,)) == []
    df, info = make_cup("u", handle_len=3, **IDEAL)
    assert _detect(df, info) is None


def test_no_breakout_means_no_signal():
    df, info = make_cup("u", breakout=False, **IDEAL)
    assert _detect(df, info) is not None
    assert generate_signals(df, P, directions=(1,)) == []


def test_handle_without_volatility_contraction_is_rejected():
    df, info = make_cup("u", handle_range=0.006, **IDEAL)
    df2, _ = make_cup("u", **IDEAL)
    assert _detect(df2, info) is not None
    assert _detect(df, info) is None


def test_deep_handle_is_rejected():
    # 押しがカップの上半分より下まで入る
    df, info = make_cup("u", handle_pull=0.035, **IDEAL)
    assert _detect(df, info) is None


def test_expiry():
    # ブレイクがハンドル完成（初めて成立した足）から signal_expiry_bars 本より後なら出ない
    df, info = make_cup("u", **IDEAL)
    assert generate_signals(df, dict(P, signal_expiry_bars=3), directions=(1,)) == []


def test_short_is_mirror_of_long():
    df, info = make_cup("u", **IDEAL)
    inv = df.copy()
    inv["open"], inv["close"] = 3 - df["open"], 3 - df["close"]
    inv["high"], inv["low"] = 3 - df["low"], 3 - df["high"]
    s_long = _detect(df, info)
    s_short = detect_cup_with_handle(inv, P, info["handle_end"], direction=-1)
    assert s_short is not None and s_short.direction == -1
    assert (s_short.i_left, s_short.i_cup, s_short.i_right) == (s_long.i_left, s_long.i_cup, s_long.i_right)
    assert np.isclose(s_short.h_right, 3 - s_long.h_right)
    assert np.isclose(s_short.l_handle, 3 - s_long.l_handle)
    sig = generate_signals(inv, P, directions=(-1,))
    assert [x.t for x in sig] == [info["breakout"]]
    assert sig[0].stop > s_short.l_handle    # ショートの損切りはハンドル高値の上


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
