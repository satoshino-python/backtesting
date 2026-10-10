"""
トレード付きのダウ理論チャート（dow_swing_chart.make_chart）用に、取引を backtesting.py の取引履歴と同じ形に直す。

- 1つのポジションの部分利確と残りの決済は、別々の行（脚）にする。
- エントリーは約定した足の始値の時刻。損切り・部分利確は、その足の中で初めて価格に触れた1分足の時刻
  （チャートの1H/4H表示で位置が合うように）。始値での成行決済はその足の始値の時刻。
- PnL は R × risk（チャートは PnL ÷ risk を R として表示する）。
"""
import numpy as np
import pandas as pd

NY = "America/New_York"


def bar_open_time(label, bar_hours, session_start_hour=17):
    """足のラベル → 始まりの時刻（NY）。日足のラベルは取引日なので、前日の NY 17:00"""
    label = pd.Timestamp(label)
    if bar_hours == 24:
        t = label.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=session_start_hour)
    else:
        t = label
    return t.tz_localize(NY, ambiguous=True, nonexistent="shift_forward")


def _first_touch(df_1min, t0, t1, level, direction, kind, spread):
    """[t0, t1) の1分足で level に初めて触れた時刻（無ければ t0）。kind="stop"/"tp"。価格は BID"""
    seg = df_1min[(df_1min.index >= t0) & (df_1min.index < t1)]
    if seg.empty:
        return t0
    if direction == 1:
        hit = seg["Low"] <= level if kind == "stop" else seg["High"] >= level
    else:
        hit = seg["High"] + spread >= level if kind == "stop" else seg["Low"] + spread <= level
    idx = np.flatnonzero(hit.to_numpy())
    return seg.index[idx[0]] if len(idx) else t0


def trades_for_chart(trades, df_1min, bar_hours, spread_by_pair, risk_display):
    rows = []
    for _, tr in trades.iterrows():
        d = int(tr["direction"])
        et = bar_open_time(tr["entry_date"], bar_hours)
        for leg in tr["legs"]:
            lab = pd.Timestamp(leg["date"])
            t0 = bar_open_time(lab, bar_hours)
            t1 = t0 + pd.Timedelta(hours=bar_hours)
            r = leg["reason"]
            if r in ("損切り", "建値ストップ"):
                level = tr["stop"] if r == "損切り" else tr["entry_price"]
                xt = _first_touch(df_1min, t0, t1, level, d, "stop", spread_by_pair[tr["pair"]])
            elif r.startswith("部分利確"):
                xt = _first_touch(df_1min, t0, t1, tr["tp1"], d, "tp", spread_by_pair[tr["pair"]])
            elif r == "期末":
                xt = t1 - pd.Timedelta(minutes=1)
            else:
                xt = t0
            rows.append(dict(
                EntryTime=et, ExitTime=max(xt, et), EntryPrice=tr["entry_price"], ExitPrice=leg["price"],
                SL=tr["stop"], TP=tr["tp1"], Size=d * leg["units"], PnL=leg["R"] * risk_display,
            ))
    return pd.DataFrame(rows)
