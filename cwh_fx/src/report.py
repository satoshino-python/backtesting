"""成績指標（仕様 12「出力する指標」）"""
import numpy as np
import pandas as pd


def max_streak(values, losing=True):
    best = cur = 0
    for v in values:
        cur = cur + 1 if (v <= 0 if losing else v > 0) else 0
        best = max(best, cur)
    return best


def trade_metrics(trades):
    """取引一覧（R 列）だけで計算できる指標"""
    if trades is None or len(trades) == 0:
        return dict(trades=0, win_rate=np.nan, avg_r=np.nan, expectancy_r=np.nan, total_r=0.0, pf=np.nan,
                    avg_win_r=np.nan, avg_loss_r=np.nan, max_losing_streak=0, max_dd_r=0.0,
                    avg_bars=np.nan, median_bars=np.nan, long=0, short=0)
    r = trades["R"].to_numpy(dtype=float)
    wins, losses = r[r > 0], r[r <= 0]
    cum = np.r_[0, np.cumsum(r)]
    return dict(
        trades=len(r), win_rate=len(wins) / len(r), avg_r=r.mean(), expectancy_r=r.mean(), total_r=r.sum(),
        pf=wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        avg_win_r=wins.mean() if len(wins) else np.nan, avg_loss_r=losses.mean() if len(losses) else np.nan,
        max_losing_streak=max_streak(r), max_dd_r=float((np.maximum.accumulate(cum) - cum).max()),
        avg_bars=float(trades["bars_held"].mean()), median_bars=float(trades["bars_held"].median()),
        long=int((trades["direction"] == 1).sum()), short=int((trades["direction"] == -1).sum()),
    )


def equity_metrics(equity, periods_per_year=260):
    """日次の資産（MTM）から: 最大DD・年率リターン・シャープ・ソルティノ"""
    e = equity["equity"].astype(float)
    if len(e) < 2:
        return dict(max_dd_pct=0.0, cagr=0.0, sharpe=np.nan, sortino=np.nan, final_equity=float(e.iloc[-1]))
    dd = 1 - e / e.cummax()
    ret = e.pct_change().dropna()
    years = (e.index[-1] - e.index[0]).days / 365.25
    cagr = (e.iloc[-1] / e.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    sd = ret.std()
    down = ret[ret < 0]
    dsd = np.sqrt((down ** 2).sum() / len(ret)) if len(ret) else 0
    return dict(
        max_dd_pct=float(dd.max()), cagr=float(cagr),
        sharpe=float(ret.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        sortino=float(ret.mean() / dsd * np.sqrt(periods_per_year)) if dsd > 0 else np.nan,
        final_equity=float(e.iloc[-1]), total_return=float(e.iloc[-1] / e.iloc[0] - 1),
    )


def all_metrics(bt):
    m = trade_metrics(bt.trades)
    m.update(equity_metrics(bt.equity))
    return m


def by_group(trades, key):
    if trades is None or len(trades) == 0:
        return pd.DataFrame(columns=["trades", "total_r", "win_rate", "avg_r"])
    g = trades.groupby(key)["R"]
    return pd.DataFrame({"trades": g.size(), "total_r": g.sum(), "win_rate": g.apply(lambda x: (x > 0).mean()),
                         "avg_r": g.mean()})


def holding_distribution(trades, bins=(0, 5, 10, 20, 40, 80, 10_000)):
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=int)
    labels = [f"{a + 1}〜{b}本" if b < 10_000 else f"{a + 1}本〜" for a, b in zip(bins[:-1], bins[1:])]
    return pd.cut(trades["bars_held"], bins=list(bins), labels=labels).value_counts().reindex(labels).fillna(0).astype(int)
