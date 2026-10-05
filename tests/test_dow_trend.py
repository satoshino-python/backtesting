"""python tests/test_dow_trend.py または pytest tests/ で実行。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dow_trend import DowConfig, atr_wilder, compute_dow_trend  # noqa: E402

FIXTURE = ROOT / "tests" / "data" / "USDJPY_2026-01_1h_bid_JST.csv"


def load_1h() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["time"], index_col="time")


def at(trend, ts):
    return int(trend.loc[pd.Timestamp(ts)])


def test_known_bars_n3():
    """チャートで目視確認した足（N=3, ATR1倍, ヒゲ判定）。"""
    t, _ = compute_dow_trend(load_1h(), DowConfig(n=3, min_swing_atr=1.0, use_wick=True))
    assert at(t, "2026-01-14 16:00") == 1     # 上昇中
    assert at(t, "2026-01-14 17:00") == -1    # 直近安値 159.082 を割り、戻り高値 159.349 < 159.454 → 下降
    assert at(t, "2026-01-15 21:00") == -1    # 下降継続
    assert at(t, "2026-01-15 22:00") == 1     # 158.733 を上抜け、押し安値 158.303 > 158.100 → 上昇


def test_known_bars_n5_delay():
    """N=5 は 1/14 12:00 の安値がスイングローにならないため、17:00 ではまだ上昇のまま。"""
    t, _ = compute_dow_trend(load_1h(), DowConfig(n=5, min_swing_atr=1.0, use_wick=True))
    assert at(t, "2026-01-14 17:00") == 1
    assert at(t, "2026-01-15 21:00") == 0


def test_no_lookahead():
    """末尾を切ったデータでも、残した範囲の判定が全期間で計算した結果と一致する。"""
    df = load_1h()
    full, _ = compute_dow_trend(df)
    for cut in (150, 300, 420):
        part, _ = compute_dow_trend(df.iloc[:cut])
        assert (part.to_numpy() == full.iloc[:cut].to_numpy()).all(), cut


def test_pivots_alternate_and_confirm_delay():
    df = load_1h()
    cfg = DowConfig(n=3)
    _, piv = compute_dow_trend(df, cfg)
    assert (piv["kind"].to_numpy()[1:] != piv["kind"].to_numpy()[:-1]).all()   # H/L が交互
    assert ((piv["confirmed_bar"] - piv["bar"]) >= cfg.n).all()                # 確定は n 本以上後


def test_min_swing_filter_reduces_pivots():
    df = load_1h()
    counts = [len(compute_dow_trend(df, DowConfig(n=3, min_swing_atr=k))[1]) for k in (0, 1, 2, 3)]
    assert counts == sorted(counts, reverse=True) and counts[0] > counts[-1]


def test_atr_wilder_known_values():
    high = np.array([10, 11, 12, 11, 13.0])
    low = np.array([9, 9.5, 10, 10, 11.0])
    close = np.array([9.5, 10.5, 11, 10.5, 12.0])
    atr = atr_wilder(high, low, close, period=3)
    tr = [1.0, 1.5, 2.0, 1.0, 2.5]                 # 最初は高値-安値、以降は True Range
    assert np.isnan(atr[:2]).all()
    assert abs(atr[2] - np.mean(tr[:3])) < 1e-12
    assert abs(atr[3] - (atr[2] * 2 + tr[3]) / 3) < 1e-12


def test_uptrend_on_synthetic_zigzag():
    """切り上げ続ける明確なジグザグでは、後半が上昇判定になる。"""
    pts = [100, 98, 104, 101, 108, 105, 112, 109, 116, 113, 120, 117]
    close = np.concatenate([np.linspace(a, b, 8, endpoint=False) for a, b in zip(pts[:-1], pts[1:])])
    df = pd.DataFrame({"Open": close, "High": close + 0.3, "Low": close - 0.3, "Close": close},
                      index=pd.date_range("2026-01-01", periods=len(close), freq="h"))
    t, _ = compute_dow_trend(df, DowConfig(n=2, min_swing_atr=0))
    assert t.iloc[-1] == 1 and (t == 1).sum() > 20


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print("ok ", fn.__name__)
    print(f"{len(tests)} passed")
