"""
ダウ理論トレンド判定（dow_trend.py / docs/DOW_TREND_SPEC.md）の確認用インタラクティブチャートを作る。

EURUSD の 1時間足・4時間足・日足・週足（NY 17:00 区切り）を HTML に埋め込み、ブラウザ側（JavaScript）で
dow_trend.py と同じアルゴリズムを実行する。ピボット幅・ATR期間・最小スイング幅(ATR倍率)などを
スライダーで変えると、スイングハイ/ローとトレンド判定がその場で再計算される。
外部ライブラリ・ネット接続は不要（HTML 1ファイルで完結）。

使い方: python dow_swing_chart.py [--symbol EURUSD] [--start 2021-01-01] [--end 2025-12-31]
出力  : dow_results/dow_swing_chart_<symbol>_<start>-<end>.html
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from main_4H_fixedSL import load_gmo_click_1min_data, get_signal_bar_label, get_trading_day_label

TEMPLATE = Path(__file__).with_name("dow_swing_chart_template.html")


def to_bars(df_1min, tf, session_start_hour=17):
    """NY 17:00 区切りで 1H/4H/D1 にまとめ、JSON 化しやすい dict にする（時刻は NY 現地時刻を UTC とみなした秒）。"""
    if tf == "D1":
        label = get_trading_day_label(df_1min.index, session_start_hour)
    elif tf == "W1":  # 金曜日で終わる週（日曜 17:00 ～ 金曜 17:00）。ラベルはその週の金曜日
        label = get_trading_day_label(df_1min.index, session_start_hour).to_period("W-FRI").end_time.normalize()
    else:
        label = get_signal_bar_label(df_1min.index, session_start_hour, int(tf[:-1]))
    b = df_1min.groupby(label).agg(Open=("Open", "first"), High=("High", "max"),
                                   Low=("Low", "min"), Close=("Close", "last"))
    t = (b.index.tz_localize(None) if b.index.tz is not None else b.index)
    secs = ((t - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(int).tolist()
    r = lambda s: [round(float(x), 5) for x in s]
    return dict(t=secs, o=r(b.Open), h=r(b.High), l=r(b.Low), c=r(b.Close))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out_dir", default="dow_results")
    a = ap.parse_args()

    ym = lambda s: int(s[:4]) * 100 + int(s[5:7])
    df, _ = load_gmo_click_1min_data(f"histData/{a.symbol}", price_side="mid",
                                     month_range=(ym(a.start), ym(a.end)))
    day = get_trading_day_label(df.index, 17)
    df = df[(day >= pd.Timestamp(a.start)) & (day <= pd.Timestamp(a.end))]

    data = {tf: to_bars(df, tf) for tf in ("1H", "4H", "D1", "W1")}
    for tf, d in data.items():
        print(f"{tf}: {len(d['t']):,} 本")
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__TITLE__", f"{a.symbol} {a.start} ~ {a.end}")
            .replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    out = Path(a.out_dir)
    out.mkdir(exist_ok=True)
    path = out / f"dow_swing_chart_{a.symbol}_{a.start.replace('-', '')}-{a.end.replace('-', '')}.html"
    path.write_text(html, encoding="utf-8")
    print(f"✅ {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
