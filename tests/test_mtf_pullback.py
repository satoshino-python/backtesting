"""python tests/test_mtf_pullback.py または pytest tests/ で実行。仕様は docs/MTF_PULLBACK_SPEC.md の「動作確認テスト」。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mtf_pullback_strategy import (  # noqa: E402
    MtfConfig, fractal_pivots, map_htf_to_ltf, mirror, prepare, run_strategy, unmirror_trades,
)

# 下位足図の折れ線（図の y 座標を 600 - y で価格にしたもの）。各区間 10 本で線形補間
PATTERN = [542, 328, 442, 190, 313, 62, 265, 170, 445, 257, 525, 500, 560]
CFG = MtfConfig(price_decimals=3, exec_1min=False)


def pattern_bars(points=PATTERN, per_leg=10):
    path = [float(points[0])]
    for a, b in zip(points[:-1], points[1:]):
        path += list(np.linspace(a, b, per_leg + 1)[1:])
    path = np.array(path)
    o, c = path[:-1], path[1:]
    idx = pd.date_range("2024-01-01", periods=len(c), freq="h")
    return pd.DataFrame({"Open": o, "High": np.maximum(o, c), "Low": np.minimum(o, c), "Close": c}, index=idx)


def fixed_context(ltf, zone_lo, zone_hi, target=600.0):
    """上位足の条件を固定（常に上昇中・底付近がゾーン）"""
    return pd.DataFrame({"Up": True, "Level": (zone_lo + zone_hi) / 2, "ZoneLo": zone_lo, "ZoneHi": zone_hi,
                         "Target": target, "ATR": 10.0}, index=ltf.index)


def test_fractal_ties_first_only():
    high = np.array([1, 2, 3, 5, 5, 3, 2, 1, 0], float)
    sh, _ = fractal_pivots(high, high, 3)
    assert list(sh) == [3]                       # 同値の2本目（位置4）は数えない


def test_pattern_reproduction():
    ltf = pattern_bars()
    trades = run_strategy(ltf, fixed_context(ltf, 40, 90), CFG)
    assert len(trades) == 1, trades
    t = trades.iloc[0]
    assert abs(t["rr_level"] - 313) < 1e-9, t["rr_level"]          # H_b。H_c（265）なら不合格
    assert abs(t["bottom"] - 62) < 1e-9
    assert abs(t["inval"] - 170) < 1e-9                             # 否定ライン = 底の後の押し安値（HL）
    green = 10 * 9                                                  # 緑丸（257）のバー位置（9本目の頂点）
    assert t["contact_pos"] <= green and t["entry_pos"] > green      # 押しの後にエントリー
    assert 250 < t["swing_low"] <= 257 + 1e-9                       # 接触安値 = 緑丸の安値
    assert t["sl"] < t["swing_low"] and t["swing_low"] - t["sl"] < 15  # 損切りは緑丸の少し下
    assert t["exit_reason"] == "TP" and t["R"] > 0


def test_random_walk_runs():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2023-01-02 00:00", periods=60 * 24 * 120, freq="min", tz="America/New_York")
    idx = idx[idx.dayofweek < 5]
    close = 1.1 + np.cumsum(rng.normal(0, 0.0002, len(idx)))
    df = pd.DataFrame({"Open": np.r_[close[0], close[:-1]], "Close": close}, index=idx)
    df["High"] = df[["Open", "Close"]].max(axis=1) + 0.0001
    df["Low"] = df[["Open", "Close"]].min(axis=1) - 0.0001
    df["Volume"] = 0
    for cfg in (MtfConfig(), MtfConfig(htf_rule="1D", ltf_hours=4, rr_target=None, support_mode="last_low")):
        for side in ("long", "short"):
            ltf, _, ctx_htf, ctx, src = prepare(df, cfg, side)
            trades = run_strategy(ltf, ctx, cfg, src, commission=0.00001)
            if not len(trades):
                continue
            assert (trades["entry"] > trades["sl"]).all() and (trades["tp"] > trades["entry"]).all()   # 反転した世界ではロング
            prev_exit = trades["exit_time"].shift(1)
            assert (trades["entry_time"][1:] > prev_exit[1:]).all()                                  # 保有中は新規なし
            if side == "short":
                real = unmirror_trades(trades)
                assert (real["sl"] > real["entry"]).all() and (real["tp"] < real["entry"]).all()


def test_short_is_mirror_of_long():
    """ショートは、反転した世界でのロングと同じ。売りの形（パターンの上下反転）で1件・実価格に戻して SL が上、TP が下"""
    ltf_long = pattern_bars()
    ltf_short = mirror(ltf_long)                              # 実際の売りの形（価格は符号反転）
    ctx = fixed_context(ltf_long, 40, 90)                     # 反転した世界（ロングの形）の上位足条件
    raw = run_strategy(mirror(ltf_short), ctx, CFG)
    t = unmirror_trades(raw).iloc[0]
    assert t["side"] == "short" and t["sl"] > t["entry"] > t["tp"]
    assert abs(t["rr_level"] - (-313)) < 1e-9 and abs(t["zone_hi"] - (-40)) < 1e-9
    assert t["R"] > 0


def test_htf_state_uses_closed_bar_only():
    """上位足の状態は、その上位足が確定した後の下位足にだけ割り当てる"""
    ltf = pattern_bars()
    labels = ltf.index.floor("4h")
    ctx_htf = pd.DataFrame({"Up": True, "Level": 1.0, "ZoneLo": 0.5, "ZoneHi": 1.5, "Target": 2.0, "ATR": 1.0},
                           index=labels.unique())
    ctx_htf.iloc[3, ctx_htf.columns.get_loc("Level")] = 99.0
    mapped = map_htf_to_ltf(ltf, ctx_htf, labels)
    hit = mapped.index[mapped["Level"] == 99.0]
    assert (hit.floor("4h") == labels.unique()[4]).all() and len(hit) == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"✅ {name}")
