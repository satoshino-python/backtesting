"""ルールE（押し目買い 30〜50%押し）を日足で実行し、取引一覧とトレード付きチャートを出力する。
使い方: python backtest/run_pullback.py [PAIR ...]   （リポジトリのルートで実行）
今回の範囲: ルールEのエントリー確認用（取引一覧・チャート）。比較ルール・感応度・資産曲線は未実装。"""
import sys, io, contextlib, datetime as dt, pathlib
import numpy as np
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import dow_swing_chart as dsc
from backtest.data_loader import load_1min, to_daily
from backtest.rules import rule_e_trades
from main_4H_fixedSL import get_trading_day_label
from main_4H_fixedSL_multi import price_decimals_for

CHART_PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "XAUUSD"]
ALL_PAIRS = ["AUDJPY", "AUDUSD", "EURAUD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPAUD", "GBPCHF",
             "GBPJPY", "GBPUSD", "NZDJPY", "NZDUSD", "USDCHF", "USDJPY", "XAUUSD"]


def ny(ts, hm="16:59"):
    return pd.Timestamp(f"{ts:%Y-%m-%d} {hm}").tz_localize("America/New_York").tz_convert("UTC")


def chart_trades(tr):
    """取引一覧 → make_chart が読む取引履歴の形式（PnL は R 換算。1R=損切り幅）"""
    return pd.DataFrame(dict(
        EntryTime=[ny(d) for d in tr.entry_date], ExitTime=[ny(d) for d in tr.exit_date],
        Size=tr.dir, EntryPrice=tr.entry_price, ExitPrice=tr.exit_price, SL=tr.stop, TP=np.nan, PnL=tr.R))


def main(pairs, out_dir, chart_pairs):
    out_dir.mkdir(parents=True, exist_ok=True)
    orig = dsc.load_trades
    all_tr = []
    for pair in pairs:
        with contextlib.redirect_stdout(io.StringIO()):
            df, _ = load_1min(pair)
        daily = to_daily(df)
        tr, trend = rule_e_trades(daily.Close)
        tr.insert(0, "pair", pair)
        all_tr.append(tr)
        print(f"{pair}: 日足 {len(daily)} 本 / 取引 {len(tr)} 件 / 合計 {tr.R.sum():+.1f}R")
        if pair not in chart_pairs:
            continue
        # 日足の判定は引け(NY17:00)で確定 → 翌取引日の背景に使う
        state = (trend["up"].astype(int) - trend["dn"].astype(int)).shift(1).fillna(0)
        fs = state.reindex(get_trading_day_label(df.index, 17)).to_numpy()
        fs = pd.Series(fs, index=df.index)
        reasons = [f"{r.reason} / L0={r.L0:.{price_decimals_for(pair)}f} H={r.H:.{price_decimals_for(pair)}f} "
                   f"D={r.D:.{price_decimals_for(pair)}f} 押し{r.retr:.0%} 保有{r.days}日" for r in tr.itertuples()]

        def patched(trades, labels, price_decimals=5, _r=reasons):
            out = orig(trades, labels, price_decimals)
            for t, r in zip(out, _r):
                t["reason"] = r
            return out
        dsc.load_trades = patched
        try:
            dsc.make_chart(df, out_dir / f"chart_{pair}.html", f"{pair} ルールE 押し目 2020-2025",
                           "2020-01-01", "2025-12-31", trades=chart_trades(tr), mode="ruleE_daily", risk=1.0,
                           price_decimals=price_decimals_for(pair), timeframes=("4H", "D1", "W1"),
                           filter_state=fs, filter_label="日足の上昇/下降トレンド（終値>MA100 かつ 126日リターン>0）",
                           ma_periods=(20, 50, 100))
        finally:
            dsc.load_trades = orig
    allt = pd.concat(all_tr, ignore_index=True)
    allt.to_csv(out_dir / "rule_e_trades.csv", index=False, encoding="utf-8-sig")
    return allt


if __name__ == "__main__":
    args = sys.argv[1:]
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    main(args or ALL_PAIRS, ROOT / "pullback_results" / f"{stamp}_ruleE_daily", args or CHART_PAIRS)
