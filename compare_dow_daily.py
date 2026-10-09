"""
dow_daily_breakout.py の自前エンジン（simulate）と、同じルールを backtesting.py の Strategy で書いたものを
同じ期間で実行し、取引履歴が一致するかを確かめる。

使い方: python compare_dow_daily.py EURUSD 2024-01-01 2024-12-31 [n=2] [min_swing_atr=1.0]
        （決済 trail / tp2 / tp3 の3通りを順に比べる）
"""
import sys
import warnings

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from dow_daily_breakout import CASH, RISK_PCT, DailyConfig, prepare_pair, run_config


class DailyDowBT(Strategy):
    tp_r = None
    trail = True

    def init(self):
        self.day_order = None

    def next(self):
        d = self.data
        o_ls, o_lsl, o_ss, o_ssl = d.NLongStop[-1], d.NLongSL[-1], d.NShortStop[-1], d.NShortSL[-1]
        new_day = bool(d.NNewDay[-1])
        # 1) この足で注文が約定した、または損切りの水準に触れて取り消しになったら、その日の注文は終わり
        if self.day_order is not None:
            side, stop, sl = self.day_order
            touched = d.Low[-1] <= sl if side > 0 else d.High[-1] >= sl
            if self.position or touched:
                self.day_order = None
        # 2) 次の足から新しい取引日: 損切りの引き上げと、その日の注文
        if new_day:
            if self.position and self.trail:
                t = self.trades[-1]
                lv = d.NTrailL[-1] if t.is_long else d.NTrailH[-1]
                if not np.isnan(lv) and (lv > t.sl if t.is_long else lv < t.sl):
                    t.sl = lv
            self.day_order = None
            if not self.position:
                if not np.isnan(o_ls):
                    self.day_order = (1, o_ls, o_lsl)
                elif not np.isnan(o_ss):
                    self.day_order = (-1, o_ss, o_ssl)
        for order in self.orders:
            if not order.is_contingent:
                order.cancel()
        if self.day_order is not None and not self.position:
            side, stop, sl = self.day_order
            dist = abs(stop - sl)
            size = int(CASH * RISK_PCT / dist)
            if size == 0:
                return
            tp = None if self.tp_r is None else stop + side * self.tp_r * dist
            if side > 0:
                self.buy(size=size, stop=stop, sl=sl, tp=tp)
            else:
                self.sell(size=size, stop=stop, sl=sl, tp=tp)


def main():
    pair, start, end = sys.argv[1:4]
    kw = dict(a.split("=") for a in sys.argv[4:])
    n, sw = int(kw.get("n", 2)), float(kw.get("min_swing_atr", 1.0))
    df, d1, day, comm, dec = prepare_pair(pair, start, end)
    all_ok = True
    for ex in ("trail", "tp2", "tp3"):
        cfg = DailyConfig(n=n, min_swing_atr=sw, exit=ex)
        mine, used, idx, sub_day = run_config(df, d1, day, comm, cfg, start, end)
        mine = mine[~mine.Open].reset_index(drop=True)
        # backtesting.py は最初の足で注文を出せないので、前日の最後の2本も渡して初日の注文を前日のうちに置かせる
        first = df.index.get_loc(idx[0])
        pre = min(first, 2)  # データの先頭が検証初日のとき（SP500）は前日の足が無い
        sub = df.iloc[first - pre:first + len(idx)][["Open", "High", "Low", "Close"]].copy()
        sub["Volume"] = 0
        sub_day = day[first - pre:first + len(idx)]
        u = used.reindex(sub_day)
        for c in ("LongStop", "LongSL", "ShortStop", "ShortSL", "TrailL", "TrailH"):
            # 足 i の next() では、次の足（i+1）の取引日に使う値を見る
            sub["N" + c] = np.r_[u[c].to_numpy(float)[1:], np.nan]
        sub["NNewDay"] = np.r_[(sub_day[1:] != sub_day[:-1]).astype(float), 0.0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bt = Backtest(sub, DailyDowBT, cash=CASH, commission=comm, margin=1 / 25, exclusive_orders=False)
            st = bt.run(tp_r=cfg.tp_r, trail=cfg.tp_r is None)
        ref = st["_trades"].reset_index(drop=True)
        cols = ["Size", "EntryPrice", "ExitPrice", "EntryTime", "ExitTime"]
        same = len(ref) == len(mine)
        if same:
            a, b = mine[cols].copy(), ref[cols].copy()
            same = ((a.Size.values == b.Size.values).all()
                    and np.allclose(a.EntryPrice, b.EntryPrice) and np.allclose(a.ExitPrice, b.ExitPrice)
                    and (pd.DatetimeIndex(a.EntryTime) == pd.DatetimeIndex(b.EntryTime)).all()
                    and (pd.DatetimeIndex(a.ExitTime) == pd.DatetimeIndex(b.ExitTime)).all()
                    and np.allclose(mine.PnL, ref.PnL, atol=1e-6 * CASH))
        all_ok &= same
        print(f"{pair} {start}~{end} {cfg.name}: 自前 {len(mine)} 件 {mine.PnL.sum():,.2f} / "
              f"backtesting.py {len(ref)} 件 {ref.PnL.sum():,.2f} → {'一致' if same else '不一致'}")
        if not same:
            print(mine[cols + ["PnL"]].head(20).to_string())
            print(ref[cols + ["PnL"]].head(20).to_string())
    print("すべて一致" if all_ok else "不一致あり")


if __name__ == "__main__":
    main()
