"""指標のテスト。python cwh_fx/tests/test_indicators.py または pytest で実行"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cwh_fx.src.indicators import atr_wilder, sma, swing_high_mask, swing_low_mask, true_range


def test_true_range_uses_previous_close():
    h, l, c = [10, 12, 11], [9, 11, 8], [9.5, 11.5, 9]
    np.testing.assert_allclose(true_range(h, l, c), [1, 2.5, 3.5])


def test_atr_wilder_seed_and_recursion():
    rng = np.random.default_rng(0)
    c = 100 + rng.normal(0, 1, 60).cumsum()
    h, l = c + rng.uniform(0.1, 1, 60), c - rng.uniform(0.1, 1, 60)
    a = atr_wilder(h, l, c, 14)
    tr = true_range(h, l, c)
    assert np.isnan(a[:13]).all()
    assert np.isclose(a[13], tr[:14].mean())
    assert np.isclose(a[14], (a[13] * 13 + tr[14]) / 14)
    assert np.isclose(a[-1], (a[-2] * 13 + tr[-1]) / 14)


def test_sma():
    s = sma(np.arange(10.0), 3)
    assert np.isnan(s[:2]).all() and s[2] == 1 and s[-1] == 8


def test_swing_marks_need_window_on_both_sides():
    h = np.array([1, 2, 3, 4, 5, 9, 5, 4, 3, 2, 1, 2, 3], dtype=float)
    m = swing_high_mask(h, 5)
    assert m[5] and m.sum() == 1          # 足5 だけ（左右5本で最高値）
    assert not swing_high_mask(h[:10], 5)[5]   # 右側が5本そろわないうちは確定しない
    assert swing_low_mask(-h, 5)[5]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
