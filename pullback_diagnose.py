"""
日足ダウ×4Hブレイク（dow_d1_h4_breakout.py）の取引に「何回目の押し/戻りの後のブレイクか」を付け、回数ごとの成績を集計する。
再計算はしない（既存の trades_all.csv にエントリー時点の回数を付けるだけ）。

使い方: python pullback_diagnose.py fast_results/20261009_0620_d1dow_h4
出力  : <同じフォルダ>/pullback_diag/（trades_with_pb.csv、by_count.csv）
"""
import contextlib
import io
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_fixedSL import resample_to_signal_bars, get_signal_bar_label
from dow_daily_breakout import prepare_pair, stats
from dow_d1_h4_breakout import D1_CFG, H4_DOW, START, END, IS_END
from pullback_count import pullback_counts


def counts_for_pair(pair):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, d1, day, comm, dec = prepare_pair(pair, START, END)
        h4 = resample_to_signal_bars(df, 17, 4)
        return pair, {k: pullback_counts(h4, d1, c, H4_DOW) for k, c in D1_CFG.items()}


def bucket(n):
    return "0" if n == 0 else "1" if n == 1 else "2-3" if n <= 3 else "4+"


def main():
    src = Path(sys.argv[1])
    out = src / "pullback_diag"
    out.mkdir(exist_ok=True)
    t = pd.read_csv(src / "trades_all.csv")
    t = t[t.d1 != "none"].copy()
    with ProcessPoolExecutor(3) as ex:
        counts = dict(ex.map(counts_for_pair, sorted(t.Pair.unique())))
    et = pd.DatetimeIndex(pd.to_datetime(t.EntryTime, utc=True)).tz_convert("America/New_York")
    t["H4Bar"] = get_signal_bar_label(et, 17, 4)
    pb, stair = [], []
    for (_, x), bar in zip(t.iterrows(), t.H4Bar):
        c = counts[x.Pair][x.d1].loc[bar]
        side = "Long" if x.Side == "L" else "Short"
        pb.append(c[f"PB{side}"])
        stair.append(c[f"Stair{side}"])
    t["PB"], t["Stair"] = pb, stair
    t["Bucket"] = t.PB.map(bucket)
    t.to_csv(out / "trades_with_pb.csv", index=False, encoding="utf-8-sig")

    rows = []
    for (cfg, b), g in t.groupby(["config", "Bucket"]):
        s = stats(g.R)
        rows.append(dict(config=cfg, bucket=b, n=s["トレード数"], win=s["勝率 [%]"], avgR=s["平均R"], R=s["合計R"],
                         PF=s["プロフィットファクター"], payoff=s["ペイオフレシオ"],
                         IS_n=int(g.IS.sum()), IS_avgR=g.loc[g.IS, "R"].mean(), IS_R=g.loc[g.IS, "R"].sum(),
                         OOS_n=int((~g.IS).sum()), OOS_avgR=g.loc[~g.IS, "R"].mean(), OOS_R=g.loc[~g.IS, "R"].sum(),
                         stair_share=g.Stair.mean()))
    r = pd.DataFrame(rows)
    r.round(4).to_csv(out / "by_count.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))
    print("\n回数の分布（全設定）:", t.PB.value_counts().sort_index().to_dict())
    print(f"結果: {out}")


if __name__ == "__main__":
    main()
