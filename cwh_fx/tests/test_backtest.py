"""決済ルール（仕様 6）とコストの扱いのテスト。python cwh_fx/tests/test_backtest.py または pytest で実行"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from cwh_fx.src.backtest import EXIT_BE, EXIT_STOP, EXIT_TIME, EXIT_TP1, EXIT_TRAIL, Position, TradeSim
from cwh_fx.src.costs import CostModel
from cwh_fx.src.sizing import FxConverter, cap_by_leverage, compute_lots
from cwh_fx.tests.synthetic import load_params

PIP = 0.0001
CFG = load_params(spread_pips={"EURUSD": 0.5}, slippage_pips=0.5, stop_slippage_pips=1.0)
COSTS = CostModel(CFG, ["EURUSD"])
SIM = TradeSim(CFG, COSTS)
D = pd.Timestamp("2024-01-02")


def _pos(direction=1, entry=1.1000, stop=1.0900, units=10_000):
    r = abs(entry - stop)
    return Position(pair="EURUSD", direction=direction, entry_i=0, entry_date=D, entry_price=entry, stop0=stop,
                    units0=units, risk_acct=r * units, tp1=entry + direction * 2 * r, atr14_entry=0.005,
                    stop=stop, units=units)


def bar(o, h, l, c):
    return dict(open=o, high=h, low=l, close=c)


def test_entry_fill_includes_spread_and_slippage():
    assert np.isclose(SIM.fill_entry("EURUSD", 1, 1.1000), 1.1000 + 0.5 * PIP + 0.5 * PIP)
    assert np.isclose(SIM.fill_entry("EURUSD", -1, 1.1000), 1.1000 - 0.5 * PIP)


def test_stop_has_priority_over_tp_in_same_bar():
    p = _pos()
    SIM.intrabar(p, bar(1.1000, 1.1300, 1.0850, 1.1100), D, None)
    assert p.units == 0 and p.legs[0]["reason"] == EXIT_STOP
    assert np.isclose(p.legs[0]["price"], 1.0900 - 1.0 * PIP)


def test_gap_through_stop_fills_at_open():
    p = _pos()
    SIM.intrabar(p, bar(1.0850, 1.0870, 1.0800, 1.0820), D, None)
    assert np.isclose(p.legs[0]["price"], 1.0850 - 1.0 * PIP)


def test_short_stop_uses_ask():
    p = _pos(direction=-1, entry=1.1000, stop=1.1100)
    # BID の高値は 1.10996 だが ASK（+0.5pips）は 1.11001 で損切りに届く
    SIM.intrabar(p, bar(1.1000, 1.10996, 1.0990, 1.0995), D, None)
    assert p.units == 0 and np.isclose(p.legs[0]["price"], 1.1100 + 1.0 * PIP)


def test_partial_take_profit_then_breakeven_from_next_bar():
    p = _pos()
    SIM.intrabar(p, bar(1.1100, 1.1210, 1.1090, 1.1150), D, None)      # 2R = 1.1200 に到達
    assert p.tp1_done and p.units == 5_000 and p.legs[0]["reason"] == EXIT_TP1
    assert np.isclose(p.legs[0]["price"], 1.1200)
    assert p.stop == 1.0900                                            # 同じ足ではまだ当初の損切り
    SIM.end_of_bar(p, bar(1.1100, 1.1210, 1.1090, 1.1150), 0.003, D)
    assert p.stop == p.entry_price                                     # 次の足から建値
    SIM.intrabar(p, bar(1.1050, 1.1060, 1.0990, 1.1000), D, None)
    assert p.units == 0 and p.legs[-1]["reason"] == EXIT_BE


def test_chandelier_is_close_based_and_ratchets():
    p = _pos()
    SIM.end_of_bar(p, bar(1.1000, 1.1150, 1.0990, 1.1140), 0.003, D)   # ライン = 1.1150 − 3×0.003 = 1.1060
    assert np.isclose(p.trail, 1.1060) and p.pending_exit is None
    SIM.end_of_bar(p, bar(1.1140, 1.1145, 1.1050, 1.1070), 0.004, D)   # ATR が増えてもラインは下げない
    assert np.isclose(p.trail, 1.1060) and p.pending_exit is None       # 安値はラインを割ったが終値は上
    SIM.end_of_bar(p, bar(1.1070, 1.1080, 1.1040, 1.1055), 0.004, D)
    assert p.pending_exit == EXIT_TRAIL
    SIM.market_exit(p, 1.1050, D, None)
    assert p.units == 0 and np.isclose(p.legs[-1]["price"], 1.1050 - 0.5 * PIP)


def test_time_stop_at_40_bars_below_1r():
    p = _pos()
    for _ in range(39):
        SIM.end_of_bar(p, bar(1.1000, 1.1010, 1.0995, 1.1005), 0.0005, D)
        p.trail = np.nan                                               # トレーリングを効かせない
    assert p.pending_exit is None
    SIM.end_of_bar(p, bar(1.1000, 1.1010, 1.0995, 1.1050), 0.0005, D)  # +0.5R < 1R
    assert p.bars_held == 40 and p.pending_exit == EXIT_TIME


def test_lot_sizing_and_conversion():
    # 口座 JPY、EURUSD、損切り 50pips、資金 1,000万円の 0.5% = 5万円、USDJPY 150 → 1pip/lot = 1,500円 → 0.66 lot
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    conv = FxConverter({"USDJPY": pd.Series([150.0, 151.0], idx), "EURUSD": pd.Series([1.10, 1.11], idx)}, "JPY")
    assert conv.rate("USD", pd.Timestamp("2024-01-02"), strict=True) == 150.0   # その日より前の終値
    assert conv.rate("USD", pd.Timestamp("2024-01-02")) == 151.0
    assert np.isclose(conv.rate("EUR", pd.Timestamp("2024-01-02")), 1.11 * 151.0)
    lots = compute_lots(10_000_000, 0.005, 0.0050, PIP, 150.0)
    assert np.isclose(lots, 0.66)
    # レバレッジ 10倍: 想定元本 ≦ 1億円。1lot = 10万EUR × 165円 = 1,650万円 → 残り枠 1億 − 9,000万 で 0.6 lot まで
    assert np.isclose(cap_by_leverage(5.0, 10_000_000, 10, 90_000_000, 165.0), 0.6)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
