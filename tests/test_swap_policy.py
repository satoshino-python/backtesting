"""
スワップ（金利調整額）の反映が抜けないようにするテスト。

1. run_fast() を呼ぶ検証スクリプトは、必ず swap= を渡す（build_swap_model の戻り値。反映しないときも swap=None と明示する）。
   新しい検証スクリプトを作ったときに反映を忘れると、このテストが落ちる。
2. swap_costs.apply_swap_to_stats() の計算（保有期間・売買の向き・単位・USDへの換算）の単体テスト。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# run_fast を呼んでいても swap を渡さなくてよいスクリプト（エンジン本体・backtesting.py との一致確認）
EXEMPT = {"fast_engine.py", "compare_fast_engine.py", "swap_costs.py"}


def _calls(text):
    """text 中の run_fast( ... ) の呼び出し部分（括弧の対応を数えて切り出す）"""
    for m in re.finditer(r"(?<![\w.])run_fast\(", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[i], 0)
            i += 1
        yield text[m.start():i]


def test_every_run_fast_call_passes_swap():
    missing = []
    for path in sorted(ROOT.glob("*.py")):
        if path.name in EXEMPT:
            continue
        for call in _calls(path.read_text(encoding="utf-8")):
            if "swap=" not in call:
                missing.append(f"{path.name}: {call[:70]}...")
    assert not missing, "run_fast に swap= が渡されていません（swap_costs.build_swap_model を使う）:\n" + "\n".join(missing)


def _trades(rows):
    return pd.DataFrame(rows, columns=["Size", "EntryTime", "ExitTime", "EntryPrice", "PnL", "ReturnPct"]).assign(
        EntryTime=lambda d: pd.to_datetime(d.EntryTime).dt.tz_localize("America/New_York"),
        ExitTime=lambda d: pd.to_datetime(d.ExitTime).dt.tz_localize("America/New_York"))


def test_interest_counts_only_rollovers_held_through():
    from swap_costs import apply_interest
    table = pd.DataFrame({"date": pd.to_datetime(["2024-01-03", "2024-01-04"]), "sell": [-10, -20], "buy": [8, 18], "days": [1, 1]})
    usdjpy = pd.Series([100.0, 100.0], index=pd.to_datetime(["2024-01-01", "2024-01-10"]).tz_localize("America/New_York"))
    # 買い 2万通貨: 1/3 17:00 と 1/4 17:00 をまたぐ → (8+18) × 2 = 52円。売り 1万通貨: 1/4 のみ → -20円。保有なし → 0
    trades = _trades([
        (20_000, "2024-01-03 10:00", "2024-01-05 10:00", 1.0, 0.0, 0.0),
        (-10_000, "2024-01-03 18:00", "2024-01-05 10:00", 1.0, 0.0, 0.0),
        (10_000, "2024-01-03 10:00", "2024-01-03 12:00", 1.0, 0.0, 0.0),
    ])
    out = apply_interest(trades, table, usdjpy, lot_units=10_000)
    assert np.allclose(out["InterestJPY"], [52.0, -20.0, 0.0])
    assert np.allclose(out["Interest"], [0.52, -0.20, 0.0])          # 100円/USD
    assert np.allclose(out["PnL"], out["PnL_Gross"] + out["Interest"])


def test_real_units_scales_non_usd_quote():
    from swap_costs import real_units
    trades = _trades([(1000, "2024-01-03 10:00", "2024-01-03 12:00", 1.0, 0.0, 0.0)])
    quote_usd = pd.Series([0.01], index=pd.to_datetime(["2024-01-01"]).tz_localize("America/New_York"))  # 1円 = 0.01 USD
    assert np.allclose(real_units(trades, quote_usd), [100_000.0])   # エンジンの枚数 ÷ クォート通貨のUSD価値
    assert np.allclose(real_units(trades, None), [1000.0])           # USD クォートは換算なし


def test_apply_swap_to_stats_none_is_noop_and_model_adjusts_equity():
    from swap_costs import SwapModel, apply_swap_to_stats
    idx = pd.date_range("2024-01-02", "2024-01-08", freq="1h", tz="America/New_York")
    stats = pd.Series({"Equity Final [$]": 10_000.0, "Return [%]": 0.0, "Max. Drawdown [%]": 0.0}, dtype=object)
    stats["_trades"] = _trades([(10_000, "2024-01-03 10:00", "2024-01-05 10:00", 1.0, 0.0, 0.0)])
    stats["_equity_curve"] = pd.DataFrame({"Equity": 10_000.0, "DrawdownPct": 0.0}, index=idx)
    assert apply_swap_to_stats(stats, None, 10_000, 0.02) is stats
    table = pd.DataFrame({"date": pd.to_datetime(["2024-01-03"]), "sell": [-10], "buy": [-100], "days": [1]})
    model = SwapModel("USDJPY", table, pd.Series([100.0], index=idx[:1]), None, 10_000.0)
    out = apply_swap_to_stats(stats, model, 10_000, 0.02)
    assert np.isclose(out["Interest [USD]"], -1.0)                   # -100円 ÷ 100
    assert np.isclose(out["Equity Final [$]"], 9_999.0)
    assert np.isclose(out["_trades"]["PnL"].iloc[0], -1.0)
    assert out["Max. Drawdown [%]"] < 0
