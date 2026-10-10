"""コストモデル（仕様 9）: スプレッド・スリッページ・スワップ"""
import warnings

from .data import pip_size_for


class CostModel:
    def __init__(self, cfg, pairs):
        self.cfg = cfg
        self.spread = {}
        for pair in pairs:
            pips = (cfg.get("spread_pips") or {}).get(pair)
            if pips is None:
                pips = 1.0
                warnings.warn(f"{pair} のスプレッドが config.yaml に無いため 1.0pips とします")
            self.spread[pair] = pips * cfg.get("spread_mult", 1.0) * pip_size_for(pair)
        self.swap = cfg.get("swap_pips_per_day") or {}
        self.warned_swap = set()

    def spread_price(self, pair):
        return self.spread[pair]

    def slippage(self, pair, stop=False):
        key = "stop_slippage_pips" if stop else "slippage_pips"
        return self.cfg[key] * self.cfg.get("slippage_mult", 1.0) * pip_size_for(pair)

    def swap_per_unit(self, pair, direction, trading_day):
        """
        取引日 trading_day の終わり（NY 17:00）のロールオーバーで、1通貨あたりに受け取る（+）/支払う（−）スワップ（決済通貨の価格単位）。
        水曜のロールオーバーは3日分（受渡日が週末をまたぐため）。
        """
        side = "long" if direction == 1 else "short"
        pips = (self.swap.get(pair) or {}).get(side)
        if pips is None:
            if (pair, side) not in self.warned_swap:
                warnings.warn(f"{pair} {side} のスワップが未設定のため 0 とします（config.yaml の swap_pips_per_day）")
                self.warned_swap.add((pair, side))
            return 0.0
        days = 3 if trading_day.dayofweek == 2 else 1
        return pips * pip_size_for(pair) * days
