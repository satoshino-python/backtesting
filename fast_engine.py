"""
スイングブレイクアウト戦略（1分足執行）を backtesting.py を使わずに高速に検証するエンジン。

対応している戦略（exit_mode）:
  - "fixed"      : SwingBreakoutStrategy1Min（main_4H_fixedSL.py）。ATR倍率の固定 SL/TP＋建値ストップ
  - "swing_trail": SwingBreakoutTrail1Min（main_4H_fixedSL_multi.py）。1時間足スイングのトレーリングストップ（TP なし）

backtesting.py は1分足1本ごとに Strategy.next() とブローカー処理を Python で回すため、
5年分（約180万本）だと1ペアあたり数分かかる。この戦略は
  - ノーポジション中: 判定足（4時間足）のラインに逆指値を置き続けるだけ
  - ポジション保有中: SL / TP / 建値ストップ・トレーリングの発動 / 残っている注文を待つだけ
なので、「何かが起きる1分足」を numpy でまとめて探し、そのバーだけを
backtesting.py（0.6.x）のブローカーと同じ手順で1本ずつ処理する。
何も起きないバーは飛ばすので、結果は backtesting.py と一致したまま約100倍速くなる。

再現している backtesting.py の約定ルール（_Broker._process_orders）:
  - 逆指値は High/Low がその価格に触れたら約定。価格は「始値と逆指値の不利な方」（窓開けは始値）
  - 注文の処理順: SL注文と成行決済（Trade.close）は先頭、TP注文は末尾に入る（同じバーで SL と TP の両方に触れたら SL 優先）
  - エントリーと同じバーの SL/TP は原則として次のバーから有効。ただし「逆指値エントリーと TP に
    同じバーで触れ、SL には触れていない」場合だけ、そのバーで TP 決済する
  - fixed: ノーポジション中に置いた反対側の逆指値は、ポジションを持っても取り消されずに残る
    （exclusive_orders=False）。触れると今のポジションを決済する
  - swing_trail: ポジションを持ったバーの next() で、反対側の逆指値を取り消す。
    新しいスイングが既に終値の反対側にあるときの Trade.close() も、同じ next() の取消し
    （contingent でない注文の取消し）で消えるため、実際には成行決済されない。
    終値がスイングの内側に戻ったバーで SL が移動する
  - 建値ストップ・トレーリングは Strategy.next() と同じく、そのバーの値で判定して次のバーから有効
  - 手数料・証拠金不足による注文取消し・一部決済・期間終了時に残ったポジション（取引履歴に含めない）

使い方（bt.run() と同じ引数。df は map_signals_to_1min() で Signal* 列を付けた1分足。
swing_trail では compute_h1_swings() の H1SH / H1SL / H1SHSeq / H1SLSeq 列も必要）:
    from fast_engine import run_fast
    stats = run_fast(df, cash=cfg.cash, commission=commission_func, margin=cfg.margin,
                     sl_atr_multiplier=..., tp_atr_multiplier=..., breakeven_trigger_r=...,
                     price_decimals=..., risk_pct=...)                      # fixed
    stats = run_fast(df, ..., exit_mode="swing_trail", initial_sl_rule="atr")  # swing_trail
    stats["_trades"]  # backtesting.py と同じ列（Entry_/Exit_ の指標列は無し）

戦略のルール（SwingBreakoutStrategy1Min / move_sl_to_breakeven() / SwingBreakoutTrail1Min）を変えたら、
ここも合わせて変えること。compare_fast_engine.py で backtesting.py との一致を確かめられる。
"""
import warnings
from copy import copy
from math import copysign

import numpy as np
import pandas as pd
from backtesting._stats import compute_stats

_SIGNAL_COLS = ("SignalSH", "SignalSL", "SignalSHValid", "SignalSLValid", "SignalATR")
_H1_COLS = ("H1SH", "H1SL", "H1SHSeq", "H1SLSeq")
EXIT_MODES = ("fixed", "swing_trail")


class _Order:
    __slots__ = ("size", "stop", "limit", "sl", "tp", "parent")

    def __init__(self, size, stop=None, limit=None, sl=None, tp=None, parent=None):
        self.size = float(size)
        self.stop = stop and float(stop)
        self.limit = limit and float(limit)
        self.sl = sl and float(sl)
        self.tp = tp and float(tp)
        self.parent = parent

    @property
    def is_long(self):
        return self.size > 0

    @property
    def is_contingent(self):
        # backtesting.py と同じく、SL/TP 注文だけが contingent（Trade.close() の成行決済注文は違う）
        p = self.parent
        return p is not None and (self is p.sl_order or self is p.tp_order)


class _Trade:
    """compute_stats() にそのまま渡せるよう、backtesting.py の Trade と同じ属性名を持たせる"""
    __slots__ = ("size", "entry_price", "entry_bar", "exit_price", "exit_bar",
                 "sl_order", "tp_order", "_commissions", "entry_time", "exit_time", "tag")

    def __init__(self, size, entry_price, entry_bar):
        self.size = size
        self.entry_price = entry_price
        self.entry_bar = entry_bar
        self.exit_price = None
        self.exit_bar = None
        self.sl_order = None
        self.tp_order = None
        self._commissions = 0
        self.tag = None

    @property
    def is_long(self):
        return self.size > 0

    @property
    def sl(self):
        return self.sl_order and self.sl_order.stop

    @property
    def tp(self):
        return self.tp_order and self.tp_order.limit

    @property
    def pl(self):
        return self.size * (self.exit_price - self.entry_price) - self._commissions

    @property
    def pl_pct(self):
        gross = copysign(1, self.size) * (self.exit_price / self.entry_price - 1)
        return gross - self._commissions / (abs(self.size) * self.entry_price)


def _change_bars(*arrays):
    """どれかの値が1本前から変わったバー（NaN → 値 も変化として数える）"""
    changed = np.zeros(len(arrays[0]), dtype=bool)
    for a in arrays:
        same = (a[1:] == a[:-1]) | (np.isnan(a[1:]) & np.isnan(a[:-1]))
        changed[1:] |= ~same
    return np.flatnonzero(changed)


class _Sim:
    def __init__(self, df, cash, commission, margin, sl_atr_multiplier, tp_atr_multiplier,
                 breakeven_trigger_r, price_decimals, risk_pct, exit_mode, initial_sl_rule):
        if exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode は {EXIT_MODES} のどれかを指定してください: {exit_mode}")
        self.mode = exit_mode
        self.O = df["Open"].to_numpy(float)
        self.H = df["High"].to_numpy(float)
        self.L = df["Low"].to_numpy(float)
        self.C = df["Close"].to_numpy(float)
        self.n = len(df)
        self.cash = self.initial_cash = float(cash)
        self.comm = commission
        self.leverage = 1 / margin
        self.trigger_r = breakeven_trigger_r if exit_mode == "fixed" else None
        self.d = price_decimals
        self.orders = []
        self.trades = []
        self.closed = []
        self.cash_events = []  # (bar, cash) 現金が変わったバーと変わった後の値（資産曲線用）

        sh, sl_line, shv, slv, atr = (df[c].to_numpy(float) for c in _SIGNAL_COLS)
        indicators = [sh, sl_line, shv, slv, atr]
        if exit_mode == "swing_trail":
            missing = [c for c in _H1_COLS if c not in df.columns]
            if missing:
                raise ValueError(f"swing_trail には compute_h1_swings() の列が必要です（無い列: {missing}）")
            self.h1sh, self.h1sl, self.h1sh_seq, self.h1sl_seq = (df[c].to_numpy(float) for c in _H1_COLS)
            indicators += [self.h1sh, self.h1sl]  # Strategy.I で登録している指標（warmup の計算に入る）
            self.h1_changes_long = _change_bars(self.h1sl, self.h1sl_seq)
            self.h1_changes_short = _change_bars(self.h1sh, self.h1sh_seq)
        # Strategy.next() は Strategy.I の指標が全て揃ってから呼ばれる（backtesting.py の warmup）
        self.start = 1 + max(int(np.isnan(a).argmin()) for a in indicators)

        # ===== 各バーの Strategy.next() が置く注文（np.float64 の round と同じ np.round） =====
        d = price_decimals
        risk_amount = float(cash) * risk_pct  # init() 時点の equity × risk_pct
        with np.errstate(invalid="ignore", divide="ignore"):
            nan_free = ~(np.isnan(sh) | np.isnan(sl_line) | np.isnan(atr))
            self.eb = np.round(sh, d)
            self.es = np.round(sl_line, d)
            if exit_mode == "fixed":
                tp_dist = np.round(atr * tp_atr_multiplier, d)
                sl_dist = np.round(atr * sl_atr_multiplier, d)
                size = np.trunc(np.nan_to_num(np.where(sl_dist > 0, risk_amount / sl_dist, 0.0))).astype(np.int64)
                # ok=False のバーでは next() が注文を取り消す前に return するため、注文はそのまま残る
                ok = nan_free & (sl_dist > 0) & (size > 0)
                self.size_b = self.size_s = size
                self.tpb = np.round(self.eb + tp_dist, d)
                self.slb = np.round(self.eb - sl_dist, d)
                self.tps = np.round(self.es - tp_dist, d)
                self.sls = np.round(self.es + sl_dist, d)
                can_b = can_s = ok
            else:
                ok = nan_free
                self.tpb = self.tps = None
                self.slb, self.size_b, can_b = self._trail_entry(True, atr * sl_atr_multiplier,
                                                                 initial_sl_rule, risk_amount)
                self.sls, self.size_s, can_s = self._trail_entry(False, atr * sl_atr_multiplier,
                                                                 initial_sl_rule, risk_amount)
            # self.sh_valid[-1] の真偽（NaN は True 扱い）
            self.place_buy = ok & can_b & (self.C < sh) & (shv != 0)
            self.place_sell = ok & can_s & (self.C > sl_line) & (slv != 0)
        self.ok = ok

        check_b = self.slb < self.eb
        check_s = self.es < self.sls
        if self.tpb is not None:
            check_b &= self.eb < self.tpb
            check_s &= self.tps < self.es
        bad = (self.place_buy & ~check_b) | (self.place_sell & ~check_s)
        bad[:self.start] = False
        if bad.any():
            raise ValueError(f"SL < エントリー < TP にならない注文があります（bar {np.flatnonzero(bad)[0]}）。"
                             "backtesting.py もここで ValueError になります")

        # ===== ノーポジション中に「エントリーの逆指値に触れる」バーの一覧 =====
        # バー j の注文は、j より前で最後に ok だったバー p（の next()）が置いたもの
        idx = np.where(ok, np.arange(self.n), -1)
        idx[:self.start] = -1
        last_ok = np.maximum.accumulate(idx)
        p = np.empty(self.n, dtype=np.int64)
        p[0] = -1
        p[1:] = last_ok[:-1]
        self.placed_at = p
        pc = np.clip(p, 0, None)
        with np.errstate(invalid="ignore"):
            hit = (p >= 0) & ((self.place_buy[pc] & (self.H >= self.eb[pc])) |
                              (self.place_sell[pc] & (self.L <= self.es[pc])))
        self.flat_events = np.flatnonzero(hit)

    def _trail_entry(self, is_long, atr_dist, rule, risk_amount):
        """SwingBreakoutTrail1Min.place_entry() の当初SL・枚数（と発注できるか）をバーごとに計算する"""
        entry = self.eb if is_long else self.es
        swing = self.h1sl if is_long else self.h1sh
        sl = entry - atr_dist if is_long else entry + atr_dist
        if rule in ("near", "far"):
            usable = ~np.isnan(swing) & ((swing < entry) if is_long else (swing > entry))
            pick_max = (rule == "near") == is_long  # 買い: near→max / far→min、売り: near→min / far→max
            sl = np.where(usable, np.fmax(sl, swing) if pick_max else np.fmin(sl, swing), sl)
        elif rule != "atr":
            raise ValueError(f'initial_sl_rule は "atr" / "near" / "far" のどれかです: {rule}')
        sl = np.round(sl, self.d)
        dist = np.abs(entry - sl)
        size = np.trunc(np.nan_to_num(np.where(dist > 0, risk_amount / dist, 0.0))).astype(np.int64)
        return sl, size, (dist > 0) & (size > 0)

    # ---------- backtesting.py の _Broker と同じ処理 ----------
    def _set_contingent(self, trade, kind, price):
        old = trade.sl_order if kind == "sl" else trade.tp_order
        if old is not None and old in self.orders:
            self.orders.remove(old)
        if kind == "sl":
            order = _Order(-trade.size, stop=price, parent=trade)
            self.orders.insert(0, order)  # SL注文は先頭（同じバーなら最初に処理される）
            trade.sl_order = order
        else:
            order = _Order(-trade.size, limit=price, parent=trade)
            self.orders.append(order)
            trade.tp_order = order

    def _open_trade(self, price, size, sl, tp, i):
        trade = _Trade(size, price, i)
        self.trades.append(trade)
        self.cash -= self.comm(size, price)
        self.cash_events.append((i, self.cash))
        if tp:
            self._set_contingent(trade, "tp", tp)
        if sl:
            self._set_contingent(trade, "sl", sl)

    def _reduce_trade(self, trade, price, size, i):
        size_left = trade.size + size
        if not size_left:
            target = trade
        else:  # 一部決済: 残りの枚数で元のトレードを続け、決済した分を別のトレードとして閉じる
            trade.size = size_left
            if trade.sl_order is not None:
                trade.sl_order.size = float(-size_left)
            if trade.tp_order is not None:
                trade.tp_order.size = float(-size_left)
            target = copy(trade)
            target.size = -size
            target.sl_order = target.tp_order = None
            self.trades.append(target)
        self._close_trade(target, price, i)

    def _close_trade(self, trade, price, i):
        self.trades.remove(trade)
        if trade.sl_order is not None and trade.sl_order in self.orders:
            self.orders.remove(trade.sl_order)
        if trade.tp_order is not None and trade.tp_order in self.orders:
            self.orders.remove(trade.tp_order)
        trade.exit_price = price
        trade.exit_bar = i
        commission = self.comm(trade.size, price)
        self.cash += trade.size * (price - trade.entry_price) - commission
        self.cash_events.append((i, self.cash))
        trade._commissions = commission + self.comm(trade.size, trade.entry_price)
        self.closed.append(trade)

    def _margin_available(self, i):
        last = self.C[i]
        size_sum = sum(int(t.size) for t in self.trades)
        upl = last * size_sum - sum(t.size * t.entry_price for t in self.trades)
        used = sum(abs(t.size) * last / self.leverage for t in self.trades)
        return max(0, self.cash + upl - used)

    def _process_orders(self, i):
        o, h, l = self.O[i], self.H[i], self.L[i]
        reprocess = False
        for order in list(self.orders):
            if order not in self.orders:
                continue
            stop_price = order.stop
            if stop_price:
                if not (h >= stop_price if order.is_long else l <= stop_price):
                    continue
                order.stop = None
            if order.limit:
                is_hit = l <= order.limit if order.is_long else h >= order.limit
                before_stop = is_hit and (order.limit <= (stop_price or -np.inf) if order.is_long
                                          else order.limit >= (stop_price or np.inf))
                if not is_hit or before_stop:
                    continue
                price = (min(stop_price or o, order.limit) if order.is_long
                         else max(stop_price or o, order.limit))
            else:
                price = o
                if stop_price:
                    price = max(price, stop_price) if order.is_long else min(price, stop_price)
            is_market = not order.limit and not stop_price

            if order.parent is not None:  # SL / TP 注文、または Trade.close() の成行決済
                trade = order.parent
                size = copysign(min(abs(trade.size), abs(order.size)), order.size)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, i)
                    if order is trade.sl_order:
                        order.stop = stop_price
                if order is not trade.sl_order and order is not trade.tp_order and order in self.orders:
                    self.orders.remove(order)
                continue

            # 新規注文（エントリーの逆指値）。反対向きのポジションがあれば先にそれを決済する
            appc = price + self.comm(order.size, price) / abs(order.size)
            need = int(order.size)
            for trade in list(self.trades):
                if trade.is_long == order.is_long:
                    continue
                if abs(need) >= abs(trade.size):
                    self._close_trade(trade, price, i)
                    need += trade.size
                else:
                    self._reduce_trade(trade, price, need, i)
                    need = 0
                if not need:
                    break
            if abs(need) * appc > self._margin_available(i) * self.leverage:
                warnings.warn(f"time={i}: Broker canceled the order due to insufficient margin",
                              UserWarning)
                self.orders.remove(order)
                continue
            if need:
                self._open_trade(price, need, order.sl, order.tp, i)
                if order.sl or order.tp:
                    if is_market:
                        reprocess = True
                    elif stop_price and not order.limit and order.tp and (
                            (order.is_long and order.tp <= h and (order.sl or -np.inf) < l) or
                            (not order.is_long and order.tp >= l and (order.sl or np.inf) > h)):
                        reprocess = True
            self.orders.remove(order)
        if reprocess:
            self._process_orders(i)

    # ---------- Strategy.next() と同じ処理 ----------
    def _strategy_next(self, i):
        if self.mode == "fixed":
            # SwingBreakoutStrategy1Min.next()
            if self.trigger_r is not None:
                self._breakeven(i)
            if not self.ok[i] or self.trades:
                return
        else:
            # SwingBreakoutTrail1Min.next()
            self._trail_stops(i)
            if self.trades:
                # 残っている反対側のエントリー注文を取り消す（SL・成行決済の注文は残す）
                self.orders = [o for o in self.orders if o.is_contingent]
                return
            if not self.ok[i]:
                return
        self.orders.clear()
        self._place_entries(i)

    def _breakeven(self, i):
        """move_sl_to_breakeven()"""
        h, l, trig = self.H[i], self.L[i], self.trigger_r
        for trade in tuple(self.trades):
            sl = trade.sl
            if sl is None:
                continue
            entry = trade.entry_price
            if trade.is_long:
                r = entry - sl
                if r > 0 and h >= entry + r * trig:
                    self._set_contingent(trade, "sl", entry)
            else:
                r = sl - entry
                if r > 0 and l <= entry - r * trig:
                    self._set_contingent(trade, "sl", entry)

    def _trail_stops(self, i):
        """SwingBreakoutTrail1Min.trail_stops()"""
        close = self.C[i]
        for trade in tuple(self.trades):
            if trade.is_long:
                new_sl, seq = self.h1sl[i], self.h1sl_seq
            else:
                new_sl, seq = self.h1sh[i], self.h1sh_seq
            if np.isnan(new_sl) or not seq[i] > seq[trade.entry_bar]:
                continue
            new_sl = np.round(new_sl, self.d)
            sl = trade.sl
            if sl is not None:
                if trade.is_long and new_sl <= sl:
                    continue
                if not trade.is_long and new_sl >= sl:
                    continue
            if (new_sl < close) if trade.is_long else (new_sl > close):
                self._set_contingent(trade, "sl", new_sl)
            else:
                # Trade.close(): 成行決済の注文（先頭に入る）。ただし直後の next() の
                # 「contingent でない注文の取消し」で消えるため、約定はしない（backtesting.py と同じ）
                self.orders.insert(0, _Order(-trade.size, parent=trade))

    def _pending_trail(self, i):
        """
        バー i の時点で「SL を動かす条件は満たしたが、終値がスイングの反対側にあるため動かせなかった」
        トレードについて、終値がスイングの内側に戻れば SL が動く。その終値の境界を返す
        （買い: Close > c_hi、売り: Close < c_lo になったバーで動く）。
        """
        c_hi, c_lo = np.inf, -np.inf
        for trade in self.trades:
            if trade.is_long:
                new_sl, seq = self.h1sl[i], self.h1sl_seq
            else:
                new_sl, seq = self.h1sh[i], self.h1sh_seq
            if np.isnan(new_sl) or not seq[i] > seq[trade.entry_bar]:
                continue
            new_sl = np.round(new_sl, self.d)
            sl = trade.sl
            if sl is not None and (new_sl <= sl if trade.is_long else new_sl >= sl):
                continue
            if trade.is_long:
                c_hi = min(c_hi, new_sl)
            else:
                c_lo = max(c_lo, new_sl)
        return c_hi, c_lo

    def _place_entries(self, p):
        tp_b = self.tpb[p] if self.tpb is not None else None
        tp_s = self.tps[p] if self.tps is not None else None
        if self.place_buy[p]:
            self.orders.append(_Order(int(self.size_b[p]), stop=self.eb[p], tp=tp_b, sl=self.slb[p]))
        if self.place_sell[p]:
            self.orders.append(_Order(-int(self.size_s[p]), stop=self.es[p], tp=tp_s, sl=self.sls[p]))

    def _step(self, i):
        self._process_orders(i)
        self._strategy_next(i)

    # ---------- 何かが起きるバーを探す ----------
    def _thresholds(self):
        """保有中: High >= hi か Low <= lo になったバーで初めて何かが起きる。成行注文があれば次のバー"""
        hi, lo = np.inf, -np.inf
        for order in self.orders:
            if order.stop:
                if order.is_long:
                    hi = min(hi, order.stop)
                else:
                    lo = max(lo, order.stop)
            elif order.limit:
                if order.is_long:
                    lo = max(lo, order.limit)
                else:
                    hi = min(hi, order.limit)
            else:
                return None  # 成行注文は次のバーで必ず約定する
        if self.trigger_r is not None:
            for trade in self.trades:
                sl = trade.sl
                if sl is None:
                    continue
                entry = trade.entry_price
                if trade.is_long:
                    r = entry - sl
                    if r > 0:
                        hi = min(hi, entry + r * self.trigger_r)
                else:
                    r = sl - entry
                    if r > 0:
                        lo = max(lo, entry - r * self.trigger_r)
        return hi, lo

    def _next_hit(self, j, hi, lo, limit, c_hi=np.inf, c_lo=-np.inf):
        """
        j 以降 limit 未満で、最初に High >= hi・Low <= lo・Close > c_hi・Close < c_lo の
        どれかになるバー（無ければ limit）
        """
        chunk = 512
        while j < limit:
            k = min(j + chunk, limit)
            m = (self.H[j:k] >= hi) | (self.L[j:k] <= lo)
            if c_hi < np.inf or c_lo > -np.inf:
                m |= (self.C[j:k] > c_hi) | (self.C[j:k] < c_lo)
            if m.any():
                return j + int(m.argmax())
            j = k
            chunk = min(chunk * 4, 1 << 20)
        return limit

    def _next_trail_bar(self, i):
        """保有中のトレードについて、1時間足スイングの値が変わる次のバー（トレーリングが動き得るバー）"""
        nxt = self.n
        for trade in self.trades:
            bars = self.h1_changes_long if trade.is_long else self.h1_changes_short
            k = int(np.searchsorted(bars, i, side="right"))
            if k < len(bars):
                nxt = min(nxt, int(bars[k]))
        return nxt

    def run(self):
        n, i = self.n, self.start
        if i >= n:
            return
        self._step(i)
        while True:
            if not self.trades and self.ok[i]:
                # ノーポジションで、バー i の next() がエントリー注文を置き直した状態
                k = int(np.searchsorted(self.flat_events, i, side="right"))
                if k >= len(self.flat_events):
                    return
                i = int(self.flat_events[k])
                self.orders.clear()
                self._place_entries(int(self.placed_at[i]))
            elif self.trades:
                th = self._thresholds()
                if th is None:
                    i += 1
                elif self.mode == "swing_trail":
                    limit = self._next_trail_bar(i) + 1
                    c_hi, c_lo = self._pending_trail(i)
                    i = min(self._next_hit(i + 1, th[0], th[1], min(limit, n), c_hi, c_lo), limit - 1)
                else:
                    i = self._next_hit(i + 1, th[0], th[1], n)
                if i >= n:
                    return
            else:
                # ノーポジションだが next() が注文を置き直さなかったバー（シグナル欠損など）は1本ずつ
                i += 1
                if i >= n:
                    return
            self._step(i)

    def equity_curve(self):
        """backtesting.py の _equity と同じ（現金 + 含み損益。warmup 前は最初の値で埋める）"""
        n = self.n
        bars = np.array([b for b, _ in self.cash_events], dtype=np.int64)
        vals = np.array([v for _, v in self.cash_events], dtype=float)
        cash_arr = np.full(n, np.nan)
        cash_arr[self.start] = self.initial_cash
        if len(bars):
            # 同じバーで複数回変わったら最後の値
            last = np.r_[bars[1:] != bars[:-1], True]
            cash_arr[bars[last]] = vals[last]
        cash_arr = pd.Series(cash_arr).ffill().to_numpy()

        size_sum = np.zeros(n + 1)
        cost_sum = np.zeros(n + 1)
        for t in self.closed + self.trades:
            end = t.exit_bar if t.exit_bar is not None else n
            size_sum[t.entry_bar] += t.size
            size_sum[end] -= t.size
            cost_sum[t.entry_bar] += t.size * t.entry_price
            cost_sum[end] -= t.size * t.entry_price
        size_sum = np.cumsum(size_sum)[:n]
        cost_sum = np.cumsum(cost_sum)[:n]
        eq = cash_arr + (self.C * size_sum - cost_sum)
        eq[:self.start] = np.nan
        return pd.Series(eq).bfill().fillna(self.cash).to_numpy()


def run_fast(df, *, cash, commission, margin, sl_atr_multiplier, price_decimals, risk_pct,
             tp_atr_multiplier=None, breakeven_trigger_r=None, exit_mode="fixed", initial_sl_rule="atr"):
    """
    exit_mode="fixed":
        Backtest(df, SwingBreakoutStrategy1Min, cash=, commission=, margin=, exclusive_orders=False)
        .run(sl_atr_multiplier=, tp_atr_multiplier=, breakeven_trigger_r=, price_decimals=, risk_pct=)
    exit_mode="swing_trail":
        Backtest(df, SwingBreakoutTrail1Min, ...).run(sl_atr_multiplier=, price_decimals=, risk_pct=, initial_sl_rule=)
    と同じ結果（stats の pd.Series）を返す。

    commission は backtesting.py と同じく関数 commission(order_size, price)、または比率（float）。
    """
    if exit_mode == "fixed" and tp_atr_multiplier is None:
        raise ValueError('exit_mode="fixed" には tp_atr_multiplier が必要です')
    if not callable(commission):
        rate = float(commission)
        commission = lambda size, price: abs(size) * price * rate  # noqa: E731
    sim = _Sim(df, cash, commission, margin, sl_atr_multiplier, tp_atr_multiplier,
               breakeven_trigger_r, price_decimals, risk_pct, exit_mode, initial_sl_rule)
    sim.run()
    if sim.trades:
        warnings.warn("Some trades remain open at the end of backtest.", UserWarning)

    index = df.index
    for t in sim.closed:
        t.entry_time = index[t.entry_bar]
        t.exit_time = index[t.exit_bar]
    equity = sim.equity_curve()
    ohlc = df[["Open", "High", "Low", "Close"]]
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = compute_stats(trades=sim.closed, equity=equity, ohlc_data=ohlc,
                              strategy_instance=None, risk_free_rate=0.0)
    # strategy_instance を渡せないため、Buy & Hold の起点だけ backtesting.py に合わせて直す
    c = ohlc["Close"].to_numpy()
    first = sim.start - 1
    stats["Buy & Hold Return [%]"] = (c[-1] - c[first]) / c[first] * 100
    name = "SwingBreakoutStrategy1Min" if exit_mode == "fixed" else "SwingBreakoutTrail1Min"
    stats["_strategy"] = f"{name}(fast_engine)"
    return stats
