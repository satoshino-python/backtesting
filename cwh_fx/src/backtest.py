"""
イベント駆動のポートフォリオ・バックテスト（仕様 5・6・7・9・10）。

1本の足（日足なら1取引日）の中の処理順:
  1. 前の足の終値で決まった成行決済（トレーリング・タイムストップ）を始値で約定
  2. 前の足の終値で出たシグナルを始値で約定（優先順位の順に、見送り条件・ポジション上限を確認）
  3. 足の中: 損切り（逆指値）→ 部分利確（指値）の順に判定。同じ足で両方に届いたら損切りだけ（保守的）
  4. 終値: チャンデリアのライン更新と判定、タイムストップの判定、スワップの計上、建値への損切り移動の反映
  5. 終値で新しいシグナルを判定 → 次の足の始値で約定

価格は BID。買いは ASK（BID + スプレッド）で約定し BID で決済、売りは BID で約定し ASK で決済する。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import pip_size_for
from .signals import Signal, generate_signals
from .sizing import FxConverter, cap_by_leverage, compute_lots, floor_lot

EXIT_STOP = "損切り"
EXIT_BE = "建値ストップ"
EXIT_TP1 = "部分利確(2R)"
EXIT_TRAIL = "トレーリング"
EXIT_TIME = "タイムストップ"
EXIT_END = "期末"


def currencies(pair):
    return pair[:3], pair[3:]


@dataclass
class Position:
    pair: str
    direction: int
    entry_i: int
    entry_date: pd.Timestamp
    entry_price: float
    stop0: float
    units0: float
    risk_acct: float
    tp1: float
    atr14_entry: float
    signal: Signal = None
    stop: float = 0.0
    units: float = 0.0
    tp1_done: bool = False
    extreme: float = np.nan
    trail: float = np.nan
    next_stop: float = None
    pending_exit: str = None
    bars_held: int = 0
    pnl_acct: float = 0.0
    swap_acct: float = 0.0
    legs: list = field(default_factory=list)

    @property
    def r_price(self):
        return abs(self.entry_price - self.stop0)


class TradeSim:
    """1ポジションの決済ルール（ポートフォリオ版とランダムエントリーの比較で共用）"""

    def __init__(self, cfg, costs: CostModel):
        self.cfg = cfg
        self.costs = costs

    def fill_entry(self, pair, direction, open_bid):
        s, slip = self.costs.spread_price(pair), self.costs.slippage(pair)
        return open_bid + s + slip if direction == 1 else open_bid - slip

    def exit_side(self, pair, direction, px):
        """決済側の価格（買いは BID、売りは ASK = BID + スプレッド）"""
        return px if direction == 1 else px + self.costs.spread_price(pair)

    def _close(self, pos, units, price, date, reason, conv):
        pnl_q = pos.direction * (price - pos.entry_price) * units
        pnl = pnl_q * (conv(currencies(pos.pair)[1], date) if conv else 1.0)
        pos.pnl_acct += pnl
        pos.units -= units
        pos.legs.append(dict(date=date, price=price, units=units, reason=reason, pnl_acct=pnl))
        return pnl

    def market_exit(self, pos, open_bid, date, conv):
        px = self.exit_side(pos.pair, pos.direction, open_bid) - pos.direction * self.costs.slippage(pos.pair)
        return self._close(pos, pos.units, px, date, pos.pending_exit, conv)

    def intrabar(self, pos, bar, date, conv):
        """足の中の損切り・部分利確。決済した損益（口座通貨）を返す"""
        d, pair = pos.direction, pos.pair
        o, h, l = (self.exit_side(pair, d, bar[k]) for k in ("open", "high", "low"))
        realized = 0.0
        hit_stop = l <= pos.stop if d == 1 else h >= pos.stop
        if hit_stop:
            gap = o if (o < pos.stop if d == 1 else o > pos.stop) else pos.stop
            px = gap - d * self.costs.slippage(pair, stop=True)
            reason = EXIT_BE if pos.tp1_done else EXIT_STOP
            return self._close(pos, pos.units, px, date, reason, conv)
        if not pos.tp1_done:
            hit_tp = h >= pos.tp1 if d == 1 else l <= pos.tp1
            if hit_tp:
                px = max(pos.tp1, o) if d == 1 else min(pos.tp1, o)
                step = self.cfg["min_lot"] * self.cfg["lot_units"]
                part = floor_lot(pos.units * self.cfg["tp1_ratio"] / step, 1) * step
                if part <= 0:
                    part = pos.units
                realized += self._close(pos, part, px, date, EXIT_TP1, conv)
                pos.tp1_done = True
                pos.next_stop = pos.entry_price      # 残りの損切りを建値へ（次の足から有効）
        return realized

    def end_of_bar(self, pos, bar, atr14, date):
        """終値でのチャンデリアとタイムストップの判定（決済は次の足の始値）"""
        d = pos.direction
        pos.bars_held += 1
        ext = bar["high"] if d == 1 else bar["low"]
        pos.extreme = ext if np.isnan(pos.extreme) else (max(pos.extreme, ext) if d == 1 else min(pos.extreme, ext))
        if np.isfinite(atr14):
            line = pos.extreme - d * self.cfg["trail_atr_mult"] * atr14
            if self.cfg.get("trail_ratchet", True) and np.isfinite(pos.trail):
                line = max(pos.trail, line) if d == 1 else min(pos.trail, line)
            pos.trail = line
        close_x = self.exit_side(pos.pair, d, bar["close"])
        if np.isfinite(pos.trail) and (close_x < pos.trail if d == 1 else close_x > pos.trail):
            pos.pending_exit = EXIT_TRAIL
        elif pos.bars_held == self.cfg["time_stop_bars"]:
            if d * (close_x - pos.entry_price) / pos.r_price < self.cfg["time_stop_min_r"]:
                pos.pending_exit = EXIT_TIME
        if pos.next_stop is not None:
            pos.stop = pos.next_stop
            pos.next_stop = None


def simulate_trade(bars, pair, entry_i, direction, stop_atr_mult, cfg, costs, atr14):
    """
    ランダムエントリーのベンチマーク用: entry_i の始値で入り、同じ決済ルールで手仕舞うまでを1単位で計算して R を返す。
    損切り = 約定価格 ∓ stop_atr_mult × ATR14_{entry_i−1}。
    """
    sim = TradeSim(cfg, costs)
    o = bars["open"].to_numpy()
    a = atr14[entry_i - 1]
    fill = sim.fill_entry(pair, direction, o[entry_i])
    stop = fill - direction * stop_atr_mult * a
    r = abs(fill - stop)
    pos = Position(pair=pair, direction=direction, entry_i=entry_i, entry_date=bars.index[entry_i], entry_price=fill,
                   stop0=stop, units0=1.0, risk_acct=r, tp1=fill + direction * cfg["tp1_r"] * r, atr14_entry=a,
                   stop=stop, units=1.0)
    cfg1 = dict(cfg, min_lot=1e-9, lot_units=1.0)
    sim = TradeSim(cfg1, costs)
    recs = bars.to_dict("records")
    for i in range(entry_i, len(bars)):
        date = bars.index[i]
        if pos.pending_exit:
            sim.market_exit(pos, recs[i]["open"], date, None)
            break
        sim.intrabar(pos, recs[i], date, None)
        if pos.units <= 1e-12:
            break
        sim.end_of_bar(pos, recs[i], atr14[i], date)
    else:
        last = len(bars) - 1
        sim._close(pos, pos.units, sim.exit_side(pair, direction, recs[last]["close"]), bars.index[last], EXIT_END, None)
    return pos.pnl_acct / r, pos.bars_held


class PortfolioBacktest:
    def __init__(self, cfg, bars_by_pair, signals_by_pair=None, log=print):
        self.cfg = cfg
        self.log = log
        start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
        # 口座通貨への換算には読み込んだ全ペアを使い、取引は cfg["pairs"]（None なら全ペア）だけで行う
        self.conv = FxConverter({p: b["close"] for p, b in bars_by_pair.items()}, cfg["account_currency"])
        self.pairs = [p for p in (cfg.get("pairs") or list(bars_by_pair)) if p in bars_by_pair]
        self.bars = {p: bars_by_pair[p][bars_by_pair[p].index <= end] for p in self.pairs}
        self.costs = CostModel(cfg, self.pairs)
        self.sim = TradeSim(cfg, self.costs)
        from .indicators import atr_wilder
        self.atr14 = {p: atr_wilder(b["high"], b["low"], b["close"], cfg["atr_short"]) for p, b in self.bars.items()}
        if signals_by_pair is None:
            signals_by_pair = {p: generate_signals(b, cfg, tuple(cfg.get("directions", (1, -1))))
                               for p, b in self.bars.items()}
        signals_by_pair = {p: v for p, v in signals_by_pair.items() if p in self.bars}
        self.signals = {}
        for p, sigs in signals_by_pair.items():
            idx = self.bars[p].index
            self.signals[p] = [s for s in sigs if s.t + 1 < len(idx) and start <= idx[s.t] <= end]
        self.start, self.end = start, end

    # ---- ポートフォリオの制約 ----
    def _blocked(self, pair, open_pos):
        cfg = self.cfg
        if len(open_pos) >= cfg["max_positions"]:
            return "ポジション数の上限"
        if cfg.get("one_position_per_pair", True) and any(p.pair == pair for p in open_pos):
            return "同じペアを保有中"
        for ccy in currencies(pair):
            if sum(ccy in currencies(p.pair) for p in open_pos) >= cfg["max_exposure_per_currency"]:
                return f"{ccy} のエクスポージャー上限"
        return None

    def _priority_key(self, s):
        if self.cfg.get("priority", "shallow") == "sma_distance":
            return -s.sma_dist_atr
        return s.depth_atr

    def _notional(self, pos, date):
        return pos.units * self.conv.rate(currencies(pos.pair)[0], date, strict=True)

    def _unrealized(self, pos, date):
        b = self.bars[pos.pair]
        i = b.index.searchsorted(date, side="right") - 1
        px = self.sim.exit_side(pos.pair, pos.direction, b["close"].iloc[i])
        return pos.direction * (px - pos.entry_price) * pos.units * self.conv.rate(currencies(pos.pair)[1], date)

    def run(self):
        cfg = self.cfg
        dates = sorted(set().union(*[b.index for b in self.bars.values()]))
        pos_of = {p: {d: i for i, d in enumerate(b.index)} for p, b in self.bars.items()}
        recs = {p: b.to_dict("records") for p, b in self.bars.items()}
        sig_at = {}
        for p, sigs in self.signals.items():
            for s in sigs:
                sig_at.setdefault((p, s.t), []).append(s)

        balance = float(cfg["initial_equity"])
        equity_prev = balance
        open_pos, trades, skipped, equity_rows = [], [], [], []
        pending = []                       # (pair, 約定する足の index, Signal)
        conv = lambda ccy, d: self.conv.rate(ccy, d)

        for date in dates:
            # 1. 成行決済
            for pos in list(open_pos):
                i = pos_of[pos.pair].get(date)
                if i is not None and pos.pending_exit:
                    balance += self.sim.market_exit(pos, recs[pos.pair][i]["open"], date, conv)
                    trades.append(self._record(pos, date))
                    open_pos.remove(pos)
            # 2. エントリー
            todays = [x for x in pending if pos_of[x[0]].get(date) == x[1]]
            pending = [x for x in pending if x not in todays]
            todays.sort(key=lambda x: self._priority_key(x[2]))
            for pair, i, s in todays:
                reason = self._blocked(pair, open_pos)
                bar = recs[pair][i]
                fill = self.sim.fill_entry(pair, s.direction, bar["open"])
                d = s.direction
                dist = d * (fill - s.stop)
                if reason is None and d * (fill - s.pivot) > cfg["max_chase_atr"] * s.atr14:
                    reason = "追いかけ（ピボットから離れすぎ）"
                if reason is None and not (cfg["stop_min_atr"] * s.atr14 <= dist <= cfg["stop_max_atr"] * s.atr14):
                    reason = f"損切り幅 {dist / s.atr14:.2f}×ATR14 が範囲外"
                lots = 0.0
                if reason is None:
                    base, quote = currencies(pair)
                    q2a = self.conv.rate(quote, date, strict=True)
                    lots = compute_lots(equity_prev, cfg["risk_per_trade"], dist, pip_size_for(pair), q2a,
                                        cfg["lot_units"], cfg["min_lot"])
                    used = sum(self._notional(p, date) for p in open_pos)
                    lots = cap_by_leverage(lots, equity_prev, cfg["max_leverage"], used,
                                           self.conv.rate(base, date, strict=True), cfg["lot_units"], cfg["min_lot"])
                    if lots < cfg["min_lot"]:
                        reason = "ロットが最小単位未満"
                if reason is not None:
                    skipped.append(dict(pair=pair, direction=d, signal_date=self.bars[pair].index[s.t],
                                        entry_date=date, reason=reason, depth_atr=s.depth_atr))
                    continue
                units = lots * cfg["lot_units"]
                r = dist
                pos = Position(pair=pair, direction=d, entry_i=i, entry_date=date, entry_price=fill, stop0=s.stop,
                               units0=units, risk_acct=r * units * q2a, tp1=fill + d * cfg["tp1_r"] * r,
                               atr14_entry=s.atr14, signal=s, stop=s.stop, units=units)
                open_pos.append(pos)
            # 3〜4. 足の中と終値
            for pos in list(open_pos):
                i = pos_of[pos.pair].get(date)
                if i is None:
                    continue
                bar = recs[pos.pair][i]
                balance += self.sim.intrabar(pos, bar, date, conv)
                if pos.units <= 1e-9:
                    trades.append(self._record(pos, date))
                    open_pos.remove(pos)
                    continue
                self.sim.end_of_bar(pos, bar, self.atr14[pos.pair][i], date)
                swap = self.costs.swap_per_unit(pos.pair, pos.direction, date) * pos.units
                if swap:
                    sw = swap * self.conv.rate(currencies(pos.pair)[1], date)
                    pos.swap_acct += sw
                    pos.pnl_acct += sw
                    balance += sw
            # 5. シグナル → 次の足で約定
            for pair in self.pairs:
                i = pos_of[pair].get(date)
                if i is not None:
                    for s in sig_at.get((pair, i), []):
                        pending.append((pair, i + 1, s))
            equity_prev = balance + sum(self._unrealized(p, date) for p in open_pos)
            if date >= self.start:
                equity_rows.append(dict(date=date, equity=equity_prev, balance=balance, open_positions=len(open_pos)))

        last = dates[-1]
        for pos in open_pos:
            i = len(self.bars[pos.pair]) - 1
            px = self.sim.exit_side(pos.pair, pos.direction, recs[pos.pair][i]["close"])
            balance += self.sim._close(pos, pos.units, px, last, EXIT_END, conv)
            trades.append(self._record(pos, last))

        self.trades = pd.DataFrame(trades)
        self.skipped = pd.DataFrame(skipped)
        self.equity = pd.DataFrame(equity_rows).set_index("date")
        return self

    def _record(self, pos, exit_date):
        s, b = pos.signal, self.bars[pos.pair]
        st = s.setup
        idx = b.index
        final = pos.legs[-1]
        return dict(
            pair=pos.pair, direction=pos.direction, side="買い" if pos.direction == 1 else "売り",
            signal_date=idx[s.t], entry_date=pos.entry_date, exit_date=exit_date,
            entry_price=pos.entry_price, stop=pos.stop0, tp1=pos.tp1, pivot=s.pivot,
            exit_price=final["price"], exit_reason=final["reason"], tp1_hit=pos.tp1_done,
            lots=pos.units0 / self.cfg["lot_units"], risk_acct=pos.risk_acct,
            pnl_acct=pos.pnl_acct, swap_acct=pos.swap_acct, R=pos.pnl_acct / pos.risk_acct,
            bars_held=pos.bars_held, stop_atr=pos.r_price / pos.atr14_entry, atr14=pos.atr14_entry,
            depth_atr=st.depth_atr, pullback_atr=st.pullback_atr, cup_len=st.cup_len, handle_len=st.handle_len,
            round_ratio=st.round_ratio, prior_rise_atr=st.prior_rise_atr, sma_dist_atr=s.sma_dist_atr,
            left_date=idx[st.i_left], cup_date=idx[st.i_cup], right_date=idx[st.i_right],
            h_left=st.h_left, l_cup=st.l_cup, h_right=st.h_right, l_handle=st.l_handle,
            legs=[dict(date=str(pd.Timestamp(l["date"])), price=round(float(l["price"]), 6), units=float(l["units"]),
                       reason=l["reason"], R=float(l["pnl_acct"] / pos.risk_acct)) for l in pos.legs],
        )


def run_portfolio(cfg, bars_by_pair, signals_by_pair=None, log=print):
    return PortfolioBacktest(cfg, bars_by_pair, signals_by_pair, log).run()
