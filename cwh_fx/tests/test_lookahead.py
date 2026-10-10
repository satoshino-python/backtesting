"""
先読みがないことの検証（仕様 12.2）。データを t で切った場合と全データを渡した場合で、t 時点の判定が一致すること。
python cwh_fx/tests/test_lookahead.py または pytest で実行。実データ（cache/ の日足）があればそれも確かめる。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from cwh_fx.src.pattern import detect_at, detect_cup_with_handle, prepare
from cwh_fx.src.signals import generate_signals
from cwh_fx.tests.synthetic import load_params, make_cup

P = load_params()


def _check_df(df, params, ts):
    for d in (1, -1):
        full = prepare(df, params, d)
        for t in ts:
            a = detect_at(full, params, t)
            b = detect_cup_with_handle(df, params, t, direction=d)
            assert (a is None) == (b is None), (d, t)
            if a is not None:
                assert a.to_dict() == b.to_dict(), (d, t)


def _check_signals_prefix(df, params, cuts):
    full = generate_signals(df, params)
    for cut in cuts:
        part = generate_signals(df.iloc[:cut + 1], params)
        want = [(s.t, s.direction, s.stop) for s in full if s.t <= cut]
        assert [(s.t, s.direction, s.stop) for s in part] == want, cut


def test_synthetic_detection_is_causal():
    df, info = make_cup("u", depth=0.06, range_=0.006)
    _check_df(df, P, range(250, len(df)))
    _check_signals_prefix(df, P, [info["handle_end"], info["breakout"], len(df) - 1])


def test_future_data_does_not_change_past_detection():
    # t より後の値をでたらめに書き換えても、t までの判定は変わらない
    df, info = make_cup("u", depth=0.06, range_=0.006)
    t = info["handle_end"]
    noisy = df.copy()
    rng = np.random.default_rng(1)
    noisy.iloc[t + 1:] = noisy.iloc[t + 1:] * (1 + rng.normal(0, 0.05, noisy.iloc[t + 1:].shape))
    for d in (1, -1):
        a = detect_at(prepare(df, P, d), P, t)
        b = detect_at(prepare(noisy, P, d), P, t)
        assert (a is None) == (b is None) and (a is None or a.to_dict() == b.to_dict())


def test_real_data_if_cached():
    from cwh_fx.src.data import CACHE_DIR
    files = sorted(Path(CACHE_DIR).glob("bars24h_*_bid.pkl")) if Path(CACHE_DIR).exists() else []
    if not files:
        print("（cache/ に日足が無いので実データの確認は省略）")
        return
    import pandas as pd
    # 実データでは成立する足が少ないので、収縮の条件を外して判定が出る足を増やして確かめる
    params = dict(P, handle_atr_contraction=99.0, right_rim_mode="min")
    for f in files[:2]:
        df = pd.read_pickle(f)
        n = len(df)
        _check_df(df, params, range(250, n, 7))
        _check_signals_prefix(df, params, [n // 2, 3 * n // 4, n - 1])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
