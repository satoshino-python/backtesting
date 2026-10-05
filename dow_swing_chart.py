"""
ダウ理論トレンド判定（dow_trend.py / docs/DOW_TREND_SPEC.md）の確認用インタラクティブチャートを作る。

EURUSD の 1時間足・4時間足・日足・週足（NY 17:00 区切り）を HTML に埋め込み、ブラウザ側（JavaScript）で
dow_trend.py と同じアルゴリズムを実行する。ピボット幅・ATR期間・最小スイング幅(ATR倍率)などを
スライダーで変えると、スイングハイ/ローとトレンド判定がその場で再計算される。
外部ライブラリ・ネット接続は不要（HTML 1ファイルで完結）。

--trades に main_4H_fixedSL_dow.py の取引履歴CSV（dow_results/trades_<期間>_<モード>.csv）を渡すと、
次の表示を追加したチャートになる。
  - トレード: エントリー（▲買い / ▼売り）・決済の位置と、その間を結ぶ線（勝ち=緑 / 負け=赤）。SL/TP の水準
  - 背景: 検証で実際に使った週足トレンド（前週末の判定。weekly_trend_<期間>.csv）
  - エントリーライン: 4時間足の Swing High/Low ライン（前後 --entry-window 本、1H/4H のみ）
  - 前/次のトレードへの移動、トレードにカーソルを合わせると詳細を表示

使い方: python dow_swing_chart.py [--symbol EURUSD] [--start 2021-01-01] [--end 2025-12-31]
        python dow_swing_chart.py --trades dow_results/trades_20210101-20251231_strict.csv
出力  : dow_results/dow_swing_chart_<symbol>_<start>-<end>.html
        （--trades 指定時は dow_results/dow_trade_chart_<symbol>_<start>-<end>_<モード>.html）
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import (
    load_gmo_click_1min_data, get_signal_bar_label, get_trading_day_label,
    resample_to_signal_bars, compute_signals, map_signals_to_1min,
)

TEMPLATE = Path(__file__).with_name("dow_swing_chart_template.html")
TIMEFRAMES = ("1H", "4H", "D1", "W1")
TREND_CODE = {"上昇": 1, "レンジ": 0, "下降": -1}


def bar_label(index_ny, tf, session_start_hour=17):
    """NY時間の DatetimeIndex の各時刻が属する足のラベル（チャートの足と同じ区切り）"""
    if tf == "D1":
        return get_trading_day_label(index_ny, session_start_hour)
    if tf == "W1":  # 金曜日で終わる週（日曜 17:00 ～ 金曜 17:00）。ラベルはその週の金曜日
        return get_trading_day_label(index_ny, session_start_hour).to_period("W-FRI").end_time.normalize()
    return get_signal_bar_label(index_ny, session_start_hour, int(tf[:-1]))


def to_seconds(index):
    """チャートの時刻（NY 現地時刻を UTC とみなした秒）"""
    t = index.tz_localize(None) if index.tz is not None else index
    return ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(int).tolist()


def to_bars(df_1min, tf, extra_cols=(), session_start_hour=17):
    """
    NY 17:00 区切りで 1H/4H/D1/W1 にまとめ、JSON 化しやすい dict と足のラベルを返す。
    extra_cols は1分足の列名→出力キーの dict で、各足の最初の1分足の値を入れる（足の中で一定の値用）。
    """
    label = bar_label(df_1min.index, tf, session_start_hour)
    agg = dict(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"))
    agg.update({key: (col, "first") for col, key in dict(extra_cols).items()})
    b = df_1min.groupby(label).agg(**agg)
    r = lambda s: [round(float(x), 5) for x in s]
    out = dict(t=to_seconds(b.index), o=r(b.Open), h=r(b.High), l=r(b.Low), c=r(b.Close))
    for key in dict(extra_cols).values():
        out[key] = [None if pd.isna(x) else round(float(x), 5) for x in b[key]]
    return out, b.index


def load_trades(path, labels, price_decimals=5):
    """取引履歴CSVを読み、各時間足でのエントリー/決済の足番号を付けた dict のリストにする"""
    t = pd.read_csv(path)
    entry = pd.DatetimeIndex(pd.to_datetime(t["EntryTime"], utc=True)).tz_convert("America/New_York")
    exit_ = pd.DatetimeIndex(pd.to_datetime(t["ExitTime"], utc=True)).tz_convert("America/New_York")
    risk = (t["EntryPrice"] - t["SL"]).abs()
    tol = 10 ** -price_decimals / 2
    trades = []
    for k, row in t.iterrows():
        long = row["Size"] > 0
        xp, ep, sl, tp = row["ExitPrice"], row["EntryPrice"], row["SL"], row["TP"]
        if not pd.isna(tp) and abs(xp - tp) <= tol:
            reason = "利確(TP)"
        elif not pd.isna(sl) and abs(xp - sl) <= tol:
            reason = "建値ストップ" if abs(sl - ep) <= tol else "損切り(SL)"
        else:
            reason = "その他（反対側の逆指値など）"
        idx = {}
        for tf, lab in labels.items():
            pos = lab.get_indexer(bar_label(entry[k:k + 1], tf).append(bar_label(exit_[k:k + 1], tf)))
            idx[tf] = [int(pos[0]), int(pos[1])]
        trades.append(dict(
            d=1 if long else -1, ep=float(ep), xp=float(xp),
            sl=None if pd.isna(sl) else float(sl), tp=None if pd.isna(tp) else float(tp),
            pnl=round(float(row["PnL"]), 2), size=int(row["Size"]),
            et=entry[k].strftime("%Y-%m-%d %H:%M"), xt=exit_[k].strftime("%Y-%m-%d %H:%M"),
            reason=reason, wk=int(row.get("Entry_Dow Trend (Weekly)", 0) or 0), idx=idx,
        ))
    return trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out_dir", default="dow_results")
    ap.add_argument("--trades", default=None, help="main_4H_fixedSL_dow.py の取引履歴CSV")
    ap.add_argument("--weekly-trend", default=None,
                    help="各週に使った週足トレンドのCSV（既定: --trades と同じフォルダの weekly_trend_<期間>.csv）")
    ap.add_argument("--entry-window", type=int, default=18, help="エントリー側 Swing の前後本数（4時間足）")
    ap.add_argument("--entry-atr", type=int, default=18, help="エントリー側 ATR 期間（4時間足）")
    ap.add_argument("--risk", type=float, default=200.0, help="1R の金額（初期資金 × risk_pct）")
    a = ap.parse_args()

    ym = lambda s: int(s[:4]) * 100 + int(s[5:7])
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    # トレード表示ありのときは、バックテストと同じくデータ全体からエントリーラインを計算する（助走期間を含める）
    month_range = (None, None) if a.trades else (ym(a.start), ym(a.end))
    df, _ = load_gmo_click_1min_data(f"histData/{a.symbol}", price_side="mid", month_range=month_range)

    extra = {}
    trades = None
    if a.trades:
        signals = compute_signals(resample_to_signal_bars(df, 17, 4), window=a.entry_window,
                                  atr_period=a.entry_atr)
        df = map_signals_to_1min(df, signals, 17, 4)
        trades_path = Path(a.trades)
        wt_path = Path(a.weekly_trend) if a.weekly_trend else trades_path.with_name(
            "weekly_trend_" + "_".join(trades_path.stem.split("_")[1:2]) + ".csv")
        wt = pd.read_csv(wt_path, index_col=0, parse_dates=True, encoding="utf-8-sig")["trend"].map(TREND_CODE)
        week = bar_label(df.index, "W1")
        df["UsedTrend"] = wt.reindex(week).to_numpy()
        extra = {"SignalSH": "esh", "SignalSL": "esl", "SignalSHValid": "eshv", "SignalSLValid": "eslv",
                 "UsedTrend": "ut"}
        print(f"ℹ️ 週足トレンド: {wt_path}")

    day = get_trading_day_label(df.index, 17)
    df = df[(day >= start) & (day <= end)]

    data, labels = {}, {}
    for tf in TIMEFRAMES:
        cols = {k: v for k, v in extra.items() if tf in ("1H", "4H") or k == "UsedTrend"}
        data[tf], labels[tf] = to_bars(df, tf, cols)
        print(f"{tf}: {len(data[tf]['t']):,} 本")

    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    period = f"{a.start.replace('-', '')}-{a.end.replace('-', '')}"
    title = f"{a.symbol} {a.start} ~ {a.end}"
    path = out / f"dow_swing_chart_{a.symbol}_{period}.html"
    if a.trades:
        mode = trades_path.stem.split("_", 2)[-1]
        trades = dict(trades=load_trades(trades_path, labels), risk=a.risk, mode=mode,
                      entryWindow=a.entry_window, source=trades_path.name)
        rs = np.array([t["pnl"] for t in trades["trades"]]) / a.risk
        print(f"ℹ️ トレード {len(rs)} 件 / 合計 {rs.sum():+.2f}R（{trades_path.name}）")
        title += f" / トレード: {mode}"
        path = out / f"dow_trade_chart_{a.symbol}_{period}_{mode}.html"

    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__TITLE__", title)
            .replace("__TRADES__", json.dumps(trades, ensure_ascii=False, separators=(",", ":")))
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    path.write_text(html, encoding="utf-8")
    print(f"✅ {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
