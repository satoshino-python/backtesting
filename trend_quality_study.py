"""
main_4H_dow_swingExit.py の戦略に「トレンドの質」フィルターを組み合わせて、全通貨ペアで比較する。

- フィルター: 週足トレンドの継続週数 / 週足 ADX / 4時間足の効率比（GRID の全組み合わせ）
- 期間を分けて評価する: IN_SAMPLE で条件を選び、OUT_OF_SAMPLE で確かめる（合わせ込み防止）
  検証は全期間で1回実行し、エントリー日で2つに分けて集計する（複利なしなので結果は同じ）
- 損益は R 倍数（損益 ÷ 1R）で合算する。1R = 初期資金 × risk_pct

結果は RESULTS_ROOT/<日時>_<RUN_LABEL>/ に保存する。
"""
import itertools
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from main_4H_dow_swingExit import CFG, prepare, run_backtest

PAIRS = ("EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDJPY")
GRID = {
    "min_trend_age": (None, 8, 16, 26),
    "min_weekly_adx": (None, 25, 35),
    "min_h4_er": (None, 0.2, 0.35),
}
IN_SAMPLE = ("2021-01-01", "2023-12-31")
OUT_OF_SAMPLE = ("2024-01-01", "2025-12-31")
MIN_IS_TRADES = 40          # 条件を選ぶときに必要な IN_SAMPLE の最低トレード数（5ペア合計）
RESULTS_ROOT = Path("quality_results")
RUN_LABEL = "age_adx_er"
WORKERS = 4


def combos():
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        yield dict(zip(keys, values))


def combo_name(c):
    f = lambda v: "-" if v is None else v
    return f"age{f(c['min_trend_age'])}_adx{f(c['min_weekly_adx'])}_er{f(c['min_h4_er'])}"


def run_pair(pair):
    cfg = replace(CFG, data_path=f"histData/{pair}", price_decimals=3 if pair.endswith("JPY") else 5)
    _, df, commission_func, _ = prepare(cfg)
    risk = cfg.cash * cfg.risk_pct
    rows = []
    for c in combos():
        _, stats = run_backtest(df, replace(cfg, **c), commission_func)
        t = stats["_trades"]
        for _, tr in t.iterrows():
            rows.append(dict(Pair=pair, Combo=combo_name(c), **c, Side="L" if tr.Size > 0 else "S",
                             EntryTime=tr.EntryTime, ExitTime=tr.ExitTime, R=tr.PnL / risk))
        print(f"{pair} {combo_name(c)}: {len(t)} trades, {t.PnL.sum() / risk:+.2f}R", flush=True)
    return rows


def r_summary(r):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return dict(n=0, win=np.nan, R=0.0, avgR=np.nan, PF=np.nan, maxDD=0.0)
    cum = np.cumsum(r)
    loss = -r[r < 0].sum()
    return dict(n=len(r), win=(r > 0).mean() * 100, R=r.sum(), avgR=r.mean(),
                PF=r[r > 0].sum() / loss if loss else np.nan,
                maxDD=float((np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]).max()))


def main():
    out_dir = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{RUN_LABEL}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "code").mkdir()
    for f in ("trend_quality_study.py", "main_4H_dow_swingExit.py"):
        shutil.copy(f, out_dir / "code" / f)
    (out_dir / "config.json").write_text(json.dumps(dict(
        base=asdict(CFG), pairs=PAIRS, grid=GRID, in_sample=IN_SAMPLE, out_of_sample=OUT_OF_SAMPLE,
        min_is_trades=MIN_IS_TRADES), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for part in ex.map(run_pair, PAIRS) for r in part]
    trades = pd.DataFrame(rows)
    trades["EntryTime"] = pd.to_datetime(trades["EntryTime"], utc=True)
    trades = trades.sort_values(["Combo", "ExitTime"])
    trades.to_csv(out_dir / "trades_all.csv", index=False)

    def period_mask(p):
        lo, hi = pd.Timestamp(p[0], tz="UTC"), pd.Timestamp(p[1], tz="UTC") + pd.Timedelta(days=1)
        return (trades.EntryTime >= lo) & (trades.EntryTime < hi)

    is_m, oos_m = period_mask(IN_SAMPLE), period_mask(OUT_OF_SAMPLE)
    table = []
    for name, g in trades.groupby("Combo", sort=False):
        row = {"Combo": name}
        for tag, m in (("IS", is_m), ("OOS", oos_m), ("ALL", is_m | oos_m)):
            s = r_summary(g.loc[m[g.index], "R"])
            row.update({f"{tag}_{k}": v for k, v in s.items()})
        for pair in PAIRS:
            gp = g[g.Pair == pair]
            row[f"IS_R_{pair}"] = gp.loc[is_m[gp.index], "R"].sum()
            row[f"OOS_R_{pair}"] = gp.loc[oos_m[gp.index], "R"].sum()
        table.append(row)
    table = pd.DataFrame(table).set_index("Combo")
    # 1件もトレードが無い組み合わせも表に残す
    table = table.reindex([combo_name(c) for c in combos()]).fillna({"IS_n": 0, "OOS_n": 0, "ALL_n": 0})
    table.to_csv(out_dir / "summary.csv", encoding="utf-8-sig")

    eligible = table[table.IS_n >= MIN_IS_TRADES]
    best = eligible.sort_values("IS_avgR", ascending=False)
    cols = ["IS_n", "IS_win", "IS_R", "IS_avgR", "IS_PF", "OOS_n", "OOS_win", "OOS_R", "OOS_avgR", "OOS_PF"]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print("\n==== 基準（フィルターなし） ====")
        print(table.loc[[combo_name(next(combos()))], cols].round(2).to_string())
        print(f"\n==== IN_SAMPLE の平均R 上位（IS {MIN_IS_TRADES} 件以上） ====")
        print(best[cols].head(10).round(2).to_string())
    print(f"\n💾 {out_dir}/")
    return out_dir


if __name__ == "__main__":
    main()
