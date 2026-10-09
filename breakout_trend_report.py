"""
breakout_trend_run.py の結果フォルダから、スマホで読めるレポート（report.html）を作る。
見た目は filter_results/trend_filter_report.html と同じ。テンプレートは breakout_trend_report_template.html。

使い方: python breakout_trend_report.py breakout_results/<日時>_<ラベル> [比較する結果フォルダ ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATE = Path(__file__).with_name("breakout_trend_report_template.html")
# 比較用: 既存のスイングブレイクアウト高速版（同じ6ペア・同じ期間。R 単位）
COMPARE = {
    "固定SL 1.5 / TP 2.5": Path("fast_results/20261006_2301_fixed_sl1.5_tp2.5/summary.csv"),
    "1Hスイング追従": Path("fast_results/20261006_2301_trailH1w5_atr_sl1.5/summary.csv"),
}


def clean(x):
    """JSON に入れられる値にする（NaN → None、numpy → Python）"""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, np.integer):
        return int(x)
    return x


def daily_cum(tr):
    s = tr.groupby(tr["ExitTime"].dt.normalize())["R"].sum().cumsum()
    return dict(t=[d.strftime("%Y-%m-%d") for d in s.index], v=s.round(2).tolist())


def hist(values, lo, hi, step):
    edges = np.arange(lo, hi + step / 2, step)
    v = np.clip(np.asarray(values, float), lo, hi - 1e-9)
    counts, _ = np.histogram(v, bins=edges)
    return dict(lo=lo, step=step, counts=counts.tolist())


def main(run_dir, extra_runs=()):
    run_dir = Path(run_dir)
    rep = json.loads((run_dir / "report_data.json").read_text(encoding="utf-8"))
    tr = pd.read_csv(run_dir / "trades_base.csv", parse_dates=["EntryTime", "ExitTime", "SignalTime"])
    pairs = rep["pairs"]

    compare = {}
    for name, path in COMPARE.items():
        if path.exists():
            c = pd.read_csv(path, encoding="utf-8-sig").set_index("通貨ペア")
            row = c.loc["合計"]
            rows = {("全体" if k == "合計" else k): dict(trades=r["トレード数"], expectancy_R=r["平均R"], total_R=r["合計R"],
                                                       pf=r["プロフィットファクター"]) for k, r in c.iterrows()}
            compare[name] = dict(total_R=row["合計R"], pf=row["プロフィットファクター"], max_dd_R=row["最大DD [R]"],
                                 trades=row["トレード数"], rows=rows)
    for d in extra_runs:   # 同じ戦略の別の実行（例: フラクタル左右3本）
        d = Path(d)
        rd = json.loads((d / "report_data.json").read_text(encoding="utf-8"))
        label = f"左右{rd['base']['fractal_n']}本"
        rows = {r["name"]: dict(trades=r["trades"], expectancy_R=r["expectancy_R"], total_R=r["total_R"], pf=r["pf"])
                for r in rd["summary"]}
        a = rows["全体"]
        mdd = next(r["max_dd_R"] for r in rd["summary"] if r["name"] == "全体")
        compare[label] = dict(total_R=a["total_R"], pf=a["pf"], max_dd_R=mdd, trades=a["trades"], rows=rows, same=True,
                              out_dir=str(d))

    winners = tr[tr.R > 0]
    hyp = dict(
        n=len(tr),
        mae_lt_05=float((tr.MAE_ATR < 0.5).mean() * 100),
        mae_ge_15=float((tr.MAE_ATR >= 1.5 - 1e-9).mean() * 100),
        win_mae_median=float(winners.MAE_ATR.median()),
        mfe_lt_1=float((tr.MFE_ATR < 1.0).mean() * 100),
        mfe_ge_15=float((tr.MFE_ATR >= 1.5).mean() * 100),
        mfe_ge_3=float((tr.MFE_ATR >= 3).mean() * 100),
        mfe_median=float(tr.MFE_ATR.median()),
    )
    data = dict(
        meta=dict(created=rep["created"], out_dir=rep["out_dir"], start=rep["start"], end=rep["end"],
                  is_end=rep["is_end"], risk_pct=rep["risk_pct"], base=rep["base"], spreads=rep["spreads"]),
        pairs=pairs, summary=rep["summary"], yearly=rep["yearly"], yearly_stats=rep["yearly_stats"],
        by_reason=rep["by_reason"], by_side=rep["by_side"], mfe=dict(n3=rep["mfe"]["n3"], n6=rep["mfe"]["n6"],
                                                                     pct=rep["mfe"]["pct"]),
        hist_r=hist(tr.R, -2, 6, 0.25), hist_mfe=hist(tr.MFE_ATR, 0, 10, 0.25),
        hist_mae=hist(tr.MAE_ATR, 0, 2.5, 0.1), hist_mae_win=hist(winners.MAE_ATR, 0, 2.5, 0.1),
        hyp=hyp, cum=daily_cum(tr), cum_pairs={p: daily_cum(tr[tr.Symbol == p]) for p in pairs},
        equity=rep["equity"], sensitivity=rep["sensitivity"], walkforward=rep["walkforward"],
        wf_params=rep["wf_params"], cost=rep["cost"], charts=rep["charts"], compare=compare,
    )
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(clean(data), ensure_ascii=False, separators=(",", ":")))
    n = rep["base"]["fractal_n"]
    if n != 3:   # 初期値（左右3本）以外の実行は、別のアーティファクトとして区別できる名前にする
        html = html.replace("ブレイクアウト・トレンドフォロー 6ペア", f"ブレイクアウト 左右{n}本 6ペア")
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"✅ {out} ({out.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
