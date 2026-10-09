"""
日足ダウ×4Hブレイク（dow_d1_h4_breakout.py）の取引に「エントリー時点で VCP（押しの縮小）が成立していたか」を付け、
成立／不成立の成績を比べる。再計算はしない。対象はエントリーラインが前後18本（h4w18）の設定。

使い方: python vcp_diagnose.py fast_results/20261009_0620_d1dow_h4
出力  : <同じフォルダ>/vcp_diag/（trades_with_vcp.csv、by_vcp.csv）
"""
import contextlib
import io
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from main_4H_fixedSL import resample_to_signal_bars, get_signal_bar_label
from dow_daily_breakout import prepare_pair, stats
from dow_d1_h4_breakout import START, END
from vcp import vcp_scan, VcpConfig


def flags_for_pair(pair):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, d1, day, comm, dec = prepare_pair(pair, START, END)
        return pair, vcp_scan(resample_to_signal_bars(df, 17, 4), VcpConfig(), dec)[0]


def main():
    src = Path(sys.argv[1])
    out = src / "vcp_diag"
    out.mkdir(exist_ok=True)
    t = pd.read_csv(src / "trades_all.csv")
    t = t[t.entry == "h4w18"].copy()
    with ProcessPoolExecutor(3) as ex:
        flags = dict(ex.map(flags_for_pair, sorted(t.Pair.unique())))
    et = pd.DatetimeIndex(pd.to_datetime(t.EntryTime, utc=True)).tz_convert("America/New_York")
    bars = get_signal_bar_label(et, 17, 4)
    side = t.Side.map({"L": "Long", "S": "Short"}).to_numpy()
    t["VCP"] = [flags[p].at[b, f"Vcp{s}"] for p, b, s in zip(t.Pair, bars, side)]
    t["NContr"] = [flags[p].at[b, f"N{s}"] for p, b, s in zip(t.Pair, bars, side)]
    t["LastATR"] = [flags[p].at[b, f"Last{s}"] for p, b, s in zip(t.Pair, bars, side)]
    t["Decreasing"] = [flags[p].at[b, f"Dec{s}"] for p, b, s in zip(t.Pair, bars, side)]
    t.to_csv(out / "trades_with_vcp.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (cfg, v), g in t.groupby(["config", "VCP"]):
        s = stats(g.R)
        rows.append(dict(config=cfg, vcp=int(v), n=s["トレード数"], win=s["勝率 [%]"], avgR=s["平均R"], R=s["合計R"],
                         PF=s["プロフィットファクター"], payoff=s["ペイオフレシオ"],
                         IS_n=int(g.IS.sum()), IS_avgR=g.loc[g.IS, "R"].mean(), OOS_n=int((~g.IS).sum()),
                         OOS_avgR=g.loc[~g.IS, "R"].mean()))
    r = pd.DataFrame(rows)
    r.round(4).to_csv(out / "by_vcp.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(r.round(2).to_string(index=False))
    print(f"結果: {out}")


if __name__ == "__main__":
    main()
