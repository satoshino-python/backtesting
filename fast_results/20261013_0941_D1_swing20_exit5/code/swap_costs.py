"""
CFD（金スポットなど）の金利調整額を、検証結果の取引履歴に反映する。

金利調整額の一覧は GMO クリック証券の「金利調整額」シート（Googleスプレッドシート）にあり、
swap_data/gold_spot_interest.csv に保存してある（更新は swap_data/README.md を参照）。

  列: date（発生日）, sell（売り）, buy（買い）, days（付与日数）。単位は 1ロット（= 1オンス）あたりの円。
      買い = 負（支払い）、売り = 正（受取り）が多いが、金利水準によって符号は変わる。
      付与日数が 0 の日（土日・休場日）は CSV に入れていない（調整額なし）。

反映のルール:
  - 発生日ごとの NY 17:00（session_start_hour）に持っているポジションに、その日の調整額が付く。
      エントリー時刻 < 発生日の NY 17:00 <= 決済時刻 のとき対象。
      水曜日の3日分などは CSV の金額（付与日数ぶんの合計）をそのまま使う。
  - 金額 = ロット数（オンス）× 発生日の金額（買いポジションは buy 列、売りポジションは sell 列）。円建て。
  - 円建ての合計を、決済時点の USDJPY（1分足の終値の仲値）で USD に直し、取引の損益（PnL）に足す。
      （損益の通貨は USD。金スポットの 1R = 初期資金 × risk_pct も USD）
  - 資産曲線には、決済の時刻に1回で反映する（保有中の日ごとの発生は決済時にまとめて計上）。
"""
from pathlib import Path

import numpy as np
import pandas as pd


def load_interest_table(csv_path):
    """金利調整額CSV（date, sell, buy, days）を読み込み、日付順に並べて返す"""
    table = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    if table["date"].duplicated().any():
        raise ValueError(f"金利調整額CSVに同じ日付が複数あります: {csv_path}")
    return table


def interest_jpy(trades, table, session_start_hour=17, tz="America/New_York"):
    """
    取引ごとの金利調整額（円）を返す（支払いは負）。trades は EntryTime / ExitTime（tz付き）/ Size の列が必要。
    """
    rollover = (table["date"] + pd.Timedelta(hours=session_start_hour)).dt.tz_localize(
        tz, ambiguous="NaT", nonexistent="shift_forward")
    ok = rollover.notna().to_numpy()
    roll_ns = rollover[ok].astype("int64").to_numpy()
    # 累積金額（先頭に 0 を置き、区間の和を差で求める）
    cum_buy = np.concatenate([[0.0], np.cumsum(table.loc[ok, "buy"].to_numpy(dtype=float))])
    cum_sell = np.concatenate([[0.0], np.cumsum(table.loc[ok, "sell"].to_numpy(dtype=float))])

    entry = pd.DatetimeIndex(trades["EntryTime"]).tz_convert(tz).asi8
    exit_ = pd.DatetimeIndex(trades["ExitTime"]).tz_convert(tz).asi8
    lo = np.searchsorted(roll_ns, entry, side="right")   # entry < T
    hi = np.searchsorted(roll_ns, exit_, side="right")   # T <= exit
    size = trades["Size"].to_numpy(dtype=float)
    per_unit = np.where(size > 0, cum_buy[hi] - cum_buy[lo], cum_sell[hi] - cum_sell[lo])
    return np.abs(size) * per_unit


def usdjpy_close_series(usdjpy_1min):
    """USDJPY の1分足（load_gmo_click_1min_data の結果）から、決済時の換算に使う終値の系列を作る"""
    return usdjpy_1min["Close"].dropna()


def apply_interest(trades, table, usdjpy_close, session_start_hour=17):
    """
    取引履歴 trades（stats["_trades"] 相当）に金利調整額を反映したコピーを返す。
    追加する列: PnL_Gross（金利調整前）, InterestJPY, UsdJpy（決済時のレート）, Interest（USD）。
    PnL は金利調整後に置き換える（ReturnPct も同じ分だけ補正）。
    """
    out = trades.copy()
    jpy = interest_jpy(out, table, session_start_hour)
    exit_time = pd.DatetimeIndex(out["ExitTime"])
    # 決済時刻以前の最新のUSDJPY（決済と同じ分のバーがあればその終値）
    idx = usdjpy_close.index.searchsorted(exit_time, side="right") - 1
    if (idx < 0).any():
        raise ValueError("決済時刻より前のUSDJPYデータがありません（USDJPYの読み込み期間を確認してください）")
    rate = usdjpy_close.to_numpy()[idx]
    usd = jpy / rate
    out["PnL_Gross"] = out["PnL"]
    out["InterestJPY"] = jpy
    out["UsdJpy"] = rate
    out["Interest"] = usd
    out["PnL"] = out["PnL_Gross"] + usd
    if "ReturnPct" in out.columns:
        notional = (out["Size"].abs() * out["EntryPrice"]).to_numpy(dtype=float)
        out["ReturnPct"] = out["ReturnPct"] + usd / notional
    return out


def cumulative_interest_curve(trades, index, risk_amount=None):
    """
    決済時刻に金利調整額（USD）を計上した累積額を、index（1分足の時刻）の各時点に対応させた Series で返す。
    risk_amount を渡すと R 単位にして返す。
    """
    step = trades.groupby("ExitTime")["Interest"].sum().sort_index().cumsum()
    curve = step.reindex(step.index.union(index)).ffill().reindex(index).fillna(0.0)
    return curve / risk_amount if risk_amount else curve
