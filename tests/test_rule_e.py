import sys, pathlib
import numpy as np
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backtest.rules import rule_e_trades

SEGS = [np.linspace(100, 100, 150), np.linspace(100, 110, 60), np.linspace(110, 104, 15),
        np.linspace(104, 120, 40), np.linspace(120, 113.5, 15), np.linspace(113.5, 130, 40)]


def _run(**kw):
    p = np.concatenate(SEGS)
    close = pd.Series(p, index=pd.bdate_range("2020-01-01", periods=len(p)))
    tr, _ = rule_e_trades(close, ma_lb=50, trend_lb=60, **kw)
    return close, tr


def test_spec_synthetic_data_gives_no_entry():
    # 指示書の人工データ(ma_lb=50, trend_lb=60)では、仕様どおりの実装だと買いは1件も入らない（指示書の期待と食い違う）。
    # 押しの底で終値が MA を割り上昇トレンド判定が外れ、戻ったときには「ATR2倍戻り→底の確定リセット」が先に起きるため。
    for lb in (3, 1):
        _, tr = _run(trigger_lb=lb)
        assert len(tr) == 0


if __name__ == "__main__":
    test_spec_synthetic_data_gives_no_entry(); print("ok")
