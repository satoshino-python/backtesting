"""
backtesting.py（0.6.6）の compute_stats() と同じ統計値を、backtesting.py を import せずに計算する。

fast_engine.run_fast() がこれを使う。出力（pd.Series の項目・_trades・_equity_curve）は
backtesting.py の stats と同じ形・同じ値になる（compare_fast_engine.py で backtesting.py と比べて確認する）。
backtesting.py 本体の backtesting/_stats.py（AGPL-3.0）を、fast_engine の取引オブジェクト用に移植したもの。
backtesting.py の版を上げて統計値の計算が変わったら、ここも合わせて直す。
"""
import numpy as np
import pandas as pd


def _data_period(index):
    """データの足の間隔（最後の100本の間隔の中央値）"""
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


def _drawdown_duration_peaks(dd):
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame("iloc").assign(prev=iloc.shift())
    df = df[df["iloc"] > df["prev"] + 1].astype(np.int64)
    if not len(df):  # ドローダウンが無い（取引なし）
        return (dd.replace(0, np.nan),) * 2
    df["duration"] = df["iloc"].map(dd.index.__getitem__) - df["prev"].map(dd.index.__getitem__)
    df["peak_dd"] = df.apply(lambda row: dd.iloc[row["prev"]:row["iloc"] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df["duration"], df["peak_dd"]


def _geometric_mean(returns):
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def trades_frame(trades):
    """取引オブジェクトのリスト → backtesting.py の stats["_trades"] と同じ列の DataFrame"""
    df = pd.DataFrame({
        "Size": [t.size for t in trades],
        "EntryBar": [t.entry_bar for t in trades],
        "ExitBar": [t.exit_bar for t in trades],
        "EntryPrice": [t.entry_price for t in trades],
        "ExitPrice": [t.exit_price for t in trades],
        "SL": [t.sl for t in trades],
        "TP": [t.tp for t in trades],
        "PnL": [t.pl for t in trades],
        "Commission": [t._commissions for t in trades],
        "ReturnPct": [t.pl_pct for t in trades],
        "EntryTime": [t.entry_time for t in trades],
        "ExitTime": [t.exit_time for t in trades],
    })
    df["Duration"] = df["ExitTime"] - df["EntryTime"]
    df["Tag"] = [t.tag for t in trades]
    return df


def compute_stats(trades, equity, ohlc_data, first_trading_bar=0, risk_free_rate=0.0, strategy=None):
    """
    backtesting._stats.compute_stats(trades, equity, ohlc_data, strategy_instance, risk_free_rate) と同じ。
    first_trading_bar は Buy & Hold の起点（backtesting.py では指標の warmup 本数 = 最初に next() が呼ばれるバー - 1）。
    strategy は stats["_strategy"] に入れる値（文字列など）。
    """
    assert -1 < risk_free_rate < 1
    index = ohlc_data.index
    equity = np.asarray(equity, dtype=float)
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = _drawdown_duration_peaks(pd.Series(dd, index=index))
    equity_df = pd.DataFrame({"Equity": equity, "DrawdownPct": dd, "DrawdownDuration": dd_dur}, index=index)

    trades_df = trades_frame(trades)
    commissions = sum(t._commissions for t in trades)
    pl = trades_df["PnL"]
    returns = trades_df["ReturnPct"]
    durations = trades_df["Duration"]
    period = _data_period(index)

    def _round_timedelta(value):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(period, "resolution_string", None) or period.resolution
        return value.ceil(resolution)

    s = {}
    s["Start"] = index[0]
    s["End"] = index[-1]
    s["Duration"] = s["End"] - s["Start"]

    have_position = np.repeat(0, len(index))
    for t in trades_df[["EntryBar", "ExitBar"]].itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1
    s["Exposure Time [%]"] = have_position.mean() * 100
    s["Equity Final [$]"] = equity[-1]
    s["Equity Peak [$]"] = equity.max()
    if commissions:
        s["Commissions [$]"] = commissions
    s["Return [%]"] = (equity[-1] - equity[0]) / equity[0] * 100
    c = ohlc_data["Close"].values
    s["Buy & Hold Return [%]"] = (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100

    gmean_day_return = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if is_datetime_index:
        freq_days = period.days
        have_weekends = index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6
        annual_trading_days = (52 if freq_days == 7 else 12 if freq_days == 31 else
                               1 if freq_days == 365 else (365 if have_weekends else 252))
        freq = {7: "W", 31: "ME", 365: "YE"}.get(freq_days, "D")
        day_returns = equity_df["Equity"].resample(freq).last().dropna().pct_change().dropna()
        gmean_day_return = _geometric_mean(day_returns)

    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
    s["Return (Ann.) [%]"] = annualized_return * 100
    s["Volatility (Ann.) [%]"] = np.sqrt(
        (day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return) ** 2) ** annual_trading_days
        - (1 + gmean_day_return) ** (2 * annual_trading_days)) * 100
    if is_datetime_index:
        time_in_years = (s["Duration"].days + s["Duration"].seconds / 86400) / 365.25
        s["CAGR [%]"] = (((s["Equity Final [$]"] / equity[0]) ** (1 / time_in_years) - 1) * 100
                         if time_in_years else np.nan)
    s["Sharpe Ratio"] = (s["Return (Ann.) [%]"] - risk_free_rate * 100) / (s["Volatility (Ann.) [%]"] or np.nan)
    with np.errstate(divide="ignore"):
        s["Sortino Ratio"] = (annualized_return - risk_free_rate) / (
            np.sqrt(np.mean(day_returns.clip(-np.inf, 0) ** 2)) * np.sqrt(annual_trading_days))
    max_dd = -np.nan_to_num(dd.max())
    s["Calmar Ratio"] = annualized_return / (-max_dd or np.nan)
    with np.errstate(divide="ignore"):
        equity_log_returns = np.log(equity[1:] / equity[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    s["Alpha [%]"] = (s["Return [%]"] - risk_free_rate * 100
                      - beta * (s["Buy & Hold Return [%]"] - risk_free_rate * 100))
    s["Beta"] = beta
    s["Max. Drawdown [%]"] = max_dd * 100
    s["Avg. Drawdown [%]"] = -dd_peaks.mean() * 100
    s["Max. Drawdown Duration"] = _round_timedelta(dd_dur.max())
    s["Avg. Drawdown Duration"] = _round_timedelta(dd_dur.mean())
    s["# Trades"] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s["Win Rate [%]"] = win_rate * 100
    s["Best Trade [%]"] = returns.max() * 100
    s["Worst Trade [%]"] = returns.min() * 100
    s["Avg. Trade [%]"] = _geometric_mean(returns) * 100
    s["Max. Trade Duration"] = _round_timedelta(durations.max())
    s["Avg. Trade Duration"] = _round_timedelta(durations.mean())
    s["Profit Factor"] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)
    s["Expectancy [%]"] = returns.mean() * 100
    s["SQN"] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    s["Kelly Criterion"] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())

    s["_strategy"] = strategy
    s["_equity_curve"] = equity_df
    s["_trades"] = trades_df
    return Stats(s, dtype=object)


class Stats(pd.Series):
    """print() したときの表示を backtesting.py の stats と同じにする"""

    def __repr__(self):
        with pd.option_context("display.max_colwidth", 20, "display.max_rows", len(self), "display.precision", 5):
            return super().fillna(np.nan).to_string(na_rep="--", dtype=True)
