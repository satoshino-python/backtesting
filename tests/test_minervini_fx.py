"""FX版ミネルヴィニ戦略（minervini_fx.py）の単体テスト: VCP の判定・未来データを見ていないこと・約定ルール・通貨強弱"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import minervini_fx as mf  # noqa: E402


def _path(points, per_leg=6):
    """頂点（価格）を直線でつないだ日足の高値=安値の系列（スイングの形を作る用）"""
    xs = [np.linspace(a, b, per_leg, endpoint=False) for a, b in zip(points[:-1], points[1:])]
    return np.concatenate(xs + [[points[-1]]])


def test_vcp_long_pivot_and_shrinking_required():
    # 高値120→安値100（幅20）→高値118→安値108（幅10）: 縮小しているのでピボット=118
    pts = [100, 120, 100, 118, 108, 112]
    h = l = _path(pts)
    atr = np.ones(len(h))
    pl, ps = mf.vcp_pivots(h + 0.0, l + 0.0, atr, window=2, n_contr=2, max_chain=500, max_age=500)
    assert pl[-1] == 118 or np.isclose(pl[-1], 118)
    assert np.isnan(ps).all()
    # 2回目の調整が1回目より大きい（20→30）なら、ピボットは出ない
    pts2 = [100, 120, 100, 130, 100, 110]
    h2 = l2 = _path(pts2)
    pl2, _ = mf.vcp_pivots(h2 + 0.0, l2 + 0.0, np.ones(len(h2)), window=2, n_contr=2, max_chain=500, max_age=500)
    assert np.isnan(pl2).all()


def test_vcp_short_is_mirror_of_long():
    pts = [100, 120, 100, 118, 108, 112]
    h = l = _path(pts)
    atr = np.ones(len(h))
    pl, ps = mf.vcp_pivots(h + 0.0, l + 0.0, atr, 2, 2, 500, 500)
    mirrored = 200 - h
    pl_m, ps_m = mf.vcp_pivots(mirrored, mirrored.copy(), atr, 2, 2, 500, 500)
    assert np.allclose(ps_m, np.where(np.isnan(pl), np.nan, 200 - pl), equal_nan=True)
    assert np.isnan(pl_m).all()


def test_vcp_does_not_look_ahead():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 600))
    h, l = c + rng.random(600), c - rng.random(600)
    atr = np.full(600, 1.0)
    full = mf.vcp_pivots(h, l, atr, 3, 2, 120, 60)
    for t in range(150, 600, 37):
        part = mf.vcp_pivots(h[:t + 1], l[:t + 1], atr[:t + 1], 3, 2, 120, 60)
        for a, b in zip(full, part):
            assert (np.isnan(a[t]) and np.isnan(b[t])) or a[t] == b[t]


def _fake_pair(o, h, l, c):
    n = len(o)
    ts = (np.arange(n) * 60_000_000_000).astype("int64")
    return SimpleNamespace(o=np.array(o, float), h=np.array(h, float), l=np.array(l, float), c=np.array(c, float),
                           ts=ts, dstart=np.array([0]), dend=np.array([n]))


def test_fill_uses_open_on_gap_and_pivot_otherwise():
    d = _fake_pair(o=[99, 99.5, 103], h=[99.5, 100.2, 104], l=[98.5, 99.4, 102.5], c=[99.5, 100, 103.5])
    i, px = mf.find_fill(d, 0, 1, 100.0)          # 2本目で100にタッチ → 約定は 100
    assert (i, px) == (1, 100.0)
    d2 = _fake_pair(o=[102, 101], h=[103, 102], l=[101, 100.5], c=[102, 101])
    i, px = mf.find_fill(d2, 0, 1, 100.0)         # 寄りから窓開け → 始値
    assert (i, px) == (0, 102)
    d3 = _fake_pair(o=[100, 99], h=[101, 100], l=[99.6, 98.0], c=[100, 99])
    i, px = mf.find_fill(d3, 0, -1, 99.0)         # ショートのピボット99: 2本目の安値 98 で約定 → 99
    assert (i, px) == (1, 99.0)


def test_exit_stop_priority_gap_and_trail():
    n = 4
    sig = pd.DataFrame({"LL": [np.nan], "HH": [np.nan]})
    # 同じ1分足でエントリー（高値100.5）と損切り（安値 97.9 <= SL 98）の両方に触れたら損切り（価格は SL）
    d = _fake_pair(o=[99.5, 99], h=[100.5, 99.2], l=[97.9, 98.9], c=[99, 99])
    i, px, why, _ = mf.simulate_exit(d, sig, 0, 0, 100.0, 1, 98.0)
    assert (i, px, why) == (0, 98.0, "損切り")
    # 窓開けで損切りを飛び越えたら始値
    d = _fake_pair(o=[100, 100.2, 96.0], h=[100.5, 100.5, 97], l=[99.8, 99.9, 95.5], c=[100, 100, 96])
    i, px, why, _ = mf.simulate_exit(d, sig, 0, 0, 100.0, 1, 98.0)
    assert (i, px, why) == (2, 96.0, "損切り")
    # 10日安値の方が近ければそちらで手仕舞い（理由は10日安値）
    sig2 = pd.DataFrame({"LL": [99.0], "HH": [np.nan]})
    d = _fake_pair(o=[100, 100, 99.5], h=[100.5, 100.5, 100], l=[99.8, 99.9, 98.9], c=[100, 100, 99])
    i, px, why, _ = mf.simulate_exit(d, sig2, 0, 0, 100.0, 1, 98.0)
    assert (i, px, why) == (2, 99.0, "10日安値")
    # ショートは逆向き
    sig3 = pd.DataFrame({"LL": [np.nan], "HH": [101.0]})
    d = _fake_pair(o=[100, 100, 100.5], h=[100.2, 100.4, 101.5], l=[99.5, 99.6, 100.2], c=[100, 100, 101])
    i, px, why, _ = mf.simulate_exit(d, sig3, 0, 0, 100.0, -1, 102.0)
    assert (i, px, why) == (2, 101.0, "10日高値")


def test_currency_strength_direction():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=200)
    data = {}
    for p in ["AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPAUD", "GBPCHF", "GBPJPY", "GBPUSD",
              "NZDJPY", "NZDUSD", "USDCHF", "USDJPY"]:
        drift = 0.002 if p == "EURUSD" else 0.0
        close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.003, len(idx))))
        data[p] = SimpleNamespace(daily=pd.DataFrame({"Close": close}, index=idx))
    s = mf.currency_strength(data, 63)
    last = s.iloc[-1]
    assert last["EUR"] > last["USD"]            # EURUSD が上がり続けた → EUR が強く USD が弱い
    assert s.iloc[:62].isna().all().all()        # 63日たまるまでは出さない
    assert abs(s.iloc[-1].sum()) < 1e-9         # ペアごとに +z と -z が対になるので、符号反転後の合計は0（6本ずつ平均なので）
