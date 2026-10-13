"""押し目買い（前回上昇幅の30〜50%押し）ルールE。仕様: 指示書 pullback_backtest_spec.md の 2章。
日足の終値ベース。参考実装に取引ログ（L0・H・D・押しの割合・手仕舞い理由）を足しただけで、売買ロジックは同じ。"""
import numpy as np
import pandas as pd


def retracement_long(p, atr, uptrend, rmin=0.30, rmax=0.50, swing_atr=2.0, min_leg_atr=4.0,
                     trigger_lb=3, max_days=40):
    """買い方向の1銘柄分。pos（0/1）と取引ログ（list of dict）を返す。売りは -p を渡す。"""
    n = len(p)
    pos = np.zeros(n)
    trades = []
    L0 = H = D = p[0]
    iL = iH = iD = 0
    state, stop, age, cur = 0, np.nan, 0, None
    for i in range(1, n):
        x, a = p[i], atr[i]
        if np.isnan(a) or np.isnan(x):
            continue
        if state == 1:
            age += 1
            reason = ("損切り(終値<D)" if x < stop else "トレンド崩れ" if not uptrend[i]
                      else "最大保有日数" if age >= max_days else None)
            if reason:
                cur.update(exit_i=i, exit_price=x, reason=reason, days=age)
                trades.append(cur)
                state, cur = 0, None
        leg = H - L0
        if state == 0 and uptrend[i] and leg >= min_leg_atr * a:
            retr = (H - D) / leg
            trig = np.nanmax(p[max(0, i - trigger_lb):i]) if i >= trigger_lb else np.inf
            if rmin <= retr <= rmax and x > trig:
                state, stop, age = 1, D, 0
                cur = dict(entry_i=i, entry_price=x, stop=D, L0=L0, H=H, D=D, retr=retr, iL=iL, iH=iH, iD=iD)
        if x > H:
            H = D = x
            iH = iD = i
        elif x < D:
            D = x
            iD = i
        if x < L0:
            L0 = H = D = x
            iL = iH = iD = i
        elif D < H - swing_atr * a and x > D + swing_atr * a:
            L0, H, D = D, x, x
            iL, iH, iD = iD, i, i
        pos[i] = state
    if cur is not None:  # データ末尾で保有中
        cur.update(exit_i=n - 1, exit_price=p[-1], reason="データ末尾", days=age)
        trades.append(cur)
    return pos, trades


def retracement_long_pivot(p, atr, uptrend, swing_n=10, rmin=0.30, rmax=0.50, min_leg_atr=4.0,
                           trigger_lb=3, max_days=40):
    """押し安値 L0 を「前後 swing_n 本の終値で最安値の足（スイングロー）」にした版（買い方向）。
    スイングローは swing_n 本後に確定する（確定前は使わない）。H = L0 以降の最高値、D = H 以降の最安値。
    終値が L0 を割ったら L0=H=D=終値 に数え直す。エントリー・手仕舞い・売り（符号反転）は retracement_long と同じ。"""
    n = len(p)
    pos = np.zeros(n)
    trades = []
    L0 = H = D = p[0]
    iL = iH = iD = 0
    state, stop, age, cur = 0, np.nan, 0, None
    for i in range(1, n):
        x, a = p[i], atr[i]
        if np.isnan(a) or np.isnan(x):
            continue
        if state == 1:
            age += 1
            reason = ("損切り(終値<D)" if x < stop else "トレンド崩れ" if not uptrend[i]
                      else "最大保有日数" if age >= max_days else None)
            if reason:
                cur.update(exit_i=i, exit_price=x, reason=reason, days=age)
                trades.append(cur)
                state, cur = 0, None
        leg = H - L0
        if state == 0 and uptrend[i] and leg >= min_leg_atr * a:
            retr = (H - D) / leg
            trig = np.nanmax(p[max(0, i - trigger_lb):i]) if i >= trigger_lb else np.inf
            if rmin <= retr <= rmax and x > trig:
                state, stop, age = 1, D, 0
                cur = dict(entry_i=i, entry_price=x, stop=D, L0=L0, H=H, D=D, retr=retr, iL=iL, iH=iH, iD=iD)
        if x > H:
            H = D = x
            iH = iD = i
        elif x < D:
            D = x
            iD = i
        if x < L0:
            L0 = H = D = x
            iL = iH = iD = i
        j = i - swing_n  # j のスイングローが今日確定する
        if j >= swing_n and j > iL and not np.isnan(p[j - swing_n:i + 1]).any() \
                and p[j] <= p[j - swing_n:i + 1].min():
            L0, iL = p[j], j
            seg = p[j:i + 1]
            iH = j + int(np.argmax(seg)); H = p[iH]
            iD = iH + int(np.argmin(p[iH:i + 1])); D = p[iD]
        pos[i] = state
    if cur is not None:
        cur.update(exit_i=n - 1, exit_price=p[-1], reason="データ末尾", days=age)
        trades.append(cur)
    return pos, trades


def rule_e_trades(close, **kw):
    """日足終値(Series)から、買い・売り両方の取引を DataFrame で返す。
    列: dir(+1/-1), entry_date, entry_price, exit_date, exit_price, stop, L0, H, D, retr, days, reason, R"""
    trend_lb, ma_lb = kw.pop("trend_lb", 126), kw.pop("ma_lb", 100)
    swing_n = kw.pop("swing_n", None)  # 指定すると L0 をスイングロー(前後 swing_n 本)で確定する版
    fn = retracement_long if swing_n is None else (lambda *a, **k: retracement_long_pivot(*a, swing_n=swing_n, **k))
    if swing_n is not None:
        kw.pop("swing_atr", None)
    px = close.astype(float)
    atr = px.diff().abs().ewm(span=20).mean().values
    ma = px.rolling(ma_lb).mean()
    mom = px / px.shift(trend_lb) - 1
    up, dn = ((px > ma) & (mom > 0)).values, ((px < ma) & (mom < 0)).values
    p = px.values
    rows = []
    for d, sign, trend in ((1, 1.0, up), (-1, -1.0, dn)):
        _, tr = fn(sign * p, atr, trend, **kw)
        for t in tr:
            rows.append(dict(
                dir=d, entry_date=px.index[t["entry_i"]], entry_price=sign * t["entry_price"],
                exit_date=px.index[t["exit_i"]], exit_price=sign * t["exit_price"], stop=sign * t["stop"],
                L0=sign * t["L0"], H=sign * t["H"], D=sign * t["D"], retr=t["retr"], days=t["days"],
                reason=t["reason"],
                L0_date=px.index[t["iL"]], H_date=px.index[t["iH"]], D_date=px.index[t["iD"]]))
    df = pd.DataFrame(rows)
    if len(df):
        risk = (df.entry_price - df.stop).abs()  # 1R = 損切り幅
        df["R"] = df.dir * (df.exit_price - df.entry_price) / risk
        df = df.sort_values("entry_date").reset_index(drop=True)
    return df, dict(up=pd.Series(up, px.index), dn=pd.Series(dn, px.index))
