"""ポジションサイジング（仕様 7）と口座通貨への換算"""
import math

import numpy as np
import pandas as pd


class FxConverter:
    """
    読み込んだ通貨ペアの終値で、任意の通貨 → 口座通貨のレートを求める（USD を経由する）。
    例) 口座 JPY で EURUSD の損益（USD）→ USDJPY で円に換算。USDCHF の損益（CHF）→ 1/USDCHF × USDJPY。
    """

    def __init__(self, closes, account_currency):
        self.acct = account_currency
        self.closes = {p: s.sort_index() for p, s in closes.items()}

    def _px(self, pair, date, strict):
        s = self.closes[pair]
        pos = s.index.searchsorted(date, side="left" if strict else "right") - 1
        if pos < 0:
            pos = 0
        return float(s.iloc[pos])

    def _to_usd(self, ccy, date, strict):
        if ccy == "USD":
            return 1.0
        if ccy + "USD" in self.closes:
            return self._px(ccy + "USD", date, strict)
        if "USD" + ccy in self.closes:
            return 1.0 / self._px("USD" + ccy, date, strict)
        raise KeyError(f"{ccy} を USD に換算できる通貨ペアがありません")

    def rate(self, ccy, date, strict=False):
        """ccy 1単位 = 口座通貨いくら。strict=True なら date より前の終値（その日の始値で使う＝未来を見ない）"""
        if ccy == self.acct:
            return 1.0
        return self._to_usd(ccy, date, strict) / self._to_usd(self.acct, date, strict)


def floor_lot(lots, min_lot):
    return math.floor(lots / min_lot + 1e-9) * min_lot


def compute_lots(equity, risk_pct, stop_distance, pip, quote_to_acct, lot_units=100_000, min_lot=0.01):
    """
    lots = risk_amount / (stop_distance_in_pips × pip_value_per_lot) を最小ロット単位で切り捨てる。
    pip_value_per_lot = lot_units × pip × （決済通貨 → 口座通貨のレート）
    """
    risk_amount = equity * risk_pct
    pips = stop_distance / pip
    pip_value = lot_units * pip * quote_to_acct
    if pips <= 0 or pip_value <= 0:
        return 0.0
    return floor_lot(risk_amount / (pips * pip_value), min_lot)


def cap_by_leverage(lots, equity, max_leverage, used_notional, base_to_acct, lot_units=100_000, min_lot=0.01):
    """保有中の想定元本と合わせて 有効証拠金 × max_leverage を超えないように切り下げる"""
    room = max_leverage * equity - used_notional
    if room <= 0:
        return 0.0
    max_lots = floor_lot(room / (lot_units * base_to_acct), min_lot)
    return min(lots, max_lots)
