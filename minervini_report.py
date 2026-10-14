"""minervini_results/<フォルダ> の trades.csv / variants.csv から report.html（アーティファクト用の1ページ）を作る。"""
import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import minervini_fx as mf
from main_minervini_fx import bootstrap_total_r

RISK = 200.0


def fmt(v, d=2):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


def tbl(df, first, cols, fmts=None):
    fmts = fmts or {}
    h = "".join(f"<th>{html.escape(c)}</th>" for c in [first] + cols)
    rows = ""
    for k, r in df.iterrows():
        tds = "".join(f"<td>{fmt(r[c], fmts.get(c, 2))}</td>" for c in cols)
        cls = ""
        rows += f"<tr><th scope='row'>{html.escape(str(k))}</th>{tds}</tr>"
    return f"<div class='scroll'><table><thead><tr>{h}</tr></thead><tbody>{rows}</tbody></table></div>"


def equity_svg(tr):
    W, H, L, B, T, Rm = 760, 300, 46, 28, 14, 135
    t = tr.sort_values("ExitTime")
    x = np.array([d.timestamp() for d in t["ExitTime"]])
    series = {"スワップ後": ("net", t["R"].cumsum().to_numpy()), "スワップ前": ("gross", t["R_Gross"].cumsum().to_numpy()),
              "ロング(後)": ("long", t["R"].where(t.Side == "Long", 0).cumsum().to_numpy()),
              "ショート(後)": ("short", t["R"].where(t.Side == "Short", 0).cumsum().to_numpy())}
    allv = np.concatenate([v for _, v in series.values()] + [[0]])
    lo, hi = np.floor(allv.min()), np.ceil(allv.max())
    x0 = pd.Timestamp("2021-01-01", tz="UTC").timestamp(); x1 = pd.Timestamp("2025-12-31", tz="UTC").timestamp()
    sx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - Rm)
    sy = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    g = ""
    step = 2 if hi - lo > 8 else 1
    for v in np.arange(lo, hi + 0.1, step):
        g += f"<line class='grid' x1='{L}' x2='{W-Rm}' y1='{sy(v):.1f}' y2='{sy(v):.1f}'/><text class='ax' x='{L-6}' y='{sy(v)+4:.1f}' text-anchor='end'>{v:.0f}</text>"
    for yr in range(2021, 2027):
        px = sx(pd.Timestamp(f"{yr}-01-01", tz="UTC").timestamp())
        g += f"<text class='ax' x='{px:.1f}' y='{H-8}' text-anchor='middle'>{yr}</text>"
    for name, (cls, v) in series.items():
        pts = [f"{sx(x0):.1f},{sy(0):.1f}"]
        for xi, vi in zip(x, v):
            pts.append(f"{sx(xi):.1f},{sy(vi):.1f}")
        g += f"<polyline class='ln {cls}' points='{' '.join(pts)}'/>"
        g += f"<text class='lb {cls}' x='{W-Rm+6}' y='{sy(v[-1])+4:.1f}'>{name} {v[-1]:+.1f}</text>"
    return f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='累積R'>{g}</svg>"


def build(folder):
    folder = Path(folder)
    tr = pd.read_csv(folder / "trades.csv")
    for c in ("EntryTime", "ExitTime"):
        tr[c] = pd.to_datetime(tr[c], utc=True).dt.tz_convert("America/New_York")
    var = pd.read_csv(folder / "variants.csv")
    r, rg = tr["R"].to_numpy(), tr["R_Gross"].to_numpy()
    nocost = ((tr["PnL_Gross"] + tr["SpreadUSD"]) / RISK).to_numpy()
    s, sg, sn = mf.overall(tr, "R"), mf.overall(tr, "R_Gross"), mf.summarize(nocost)
    lo, hi = bootstrap_total_r(r)
    se = r.std(ddof=1) / np.sqrt(len(r))
    conc = mf.concurrency(tr)
    by_side, by_pair, by_year = mf.table(tr, "Side"), mf.table(tr, "Pair"), mf.table(tr, "Year")
    tr["期間"] = np.where(tr["Year"] <= 2023, "2021-2023", "2024-2025")
    by_per = mf.table(tr, "期間")
    gs = mf.table(tr, "Side", "R_Gross")["合計R"]
    by_side["合計R(スワップ前)"] = gs
    by_reason = tr.groupby("ExitReason").agg(件数=("R", "size"), 合計R=("R", "sum"), 平均R=("R", "mean"), 保有日数=("ExitDay", lambda x: 0))
    hold = (tr["ExitTime"] - tr["EntryTime"]).dt.total_seconds() / 86400
    by_reason["保有日数"] = hold.groupby(tr["ExitReason"]).mean()
    near = (tr["TrailATR"] < 2).mean() * 100
    swap_r = tr["Interest"].sum() / RISK
    spread_r = tr["SpreadUSD"].sum() / RISK
    cols = ["トレード数", "合計R", "PF", "勝率[%]", "平均R", "最大DD(R)"]
    pair_files = sorted(folder.glob("dow_trade_chart_*_minervini.html"))
    links = "".join(f"<li><a href='{f.name}'>{f.name.split('_')[3]}</a> <span class='mut'>{int(by_pair.loc[f.name.split('_')[3], 'トレード数'])}件 / {by_pair.loc[f.name.split('_')[3], '合計R']:+.2f}R</span></li>" for f in pair_files)
    vcols = ["トレード数", "合計R(スワップ後)", "合計R(スワップ前)", "合計R(コスト前)", "PF", "勝率[%]", "最大DD(R)"]
    vt = var.set_index("バリエーション")
    vals = dict(
        n=s["トレード数"], net=f"{s['合計R']:+.2f}", gross=f"{sg['合計R']:+.2f}", nocost=f"{sn['合計R']:+.2f}",
        pf=fmt(s["PF"]), pfg=fmt(sg["PF"]), dd=fmt(s["最大DD(R)"]), ddg=fmt(sg["最大DD(R)"]), win=fmt(s["勝率[%]"], 0),
        avg=fmt(s["平均R"]), se=fmt(se), lo=f"{lo:+.1f}", hi=f"{hi:+.1f}", swap=f"{swap_r:+.2f}", spread=f"{spread_r:.2f}",
        eq=equity_svg(tr), t_side=tbl(by_side, "向き", cols + ["合計R(スワップ前)"]), t_year=tbl(by_year, "決済年", cols),
        t_per=tbl(by_per, "期間", cols), t_pair=tbl(by_pair, "ペア", cols),
        t_reason=tbl(by_reason, "決済理由", ["件数", "合計R", "平均R", "保有日数"], {"件数": 0, "保有日数": 1}),
        near=f"{near:.0f}", conc1=conc["最大同時保有"], conc2=conc["同通貨同方向の最大"], links=links,
        t_var=tbl(vt, "バリエーション", vcols, {"トレード数": 0}), gap=f"{(tr['GapATR'] > 0.05).sum()}",
        openn=int((tr.ExitReason == "期間終了").sum()),
    )
    page = TEMPLATE
    for k, v in vals.items():
        page = page.replace(f"__{k}__", str(v))
    (folder / "report.html").write_text(page, encoding="utf-8")
    return folder / "report.html"


TEMPLATE = open(Path(__file__).with_name("minervini_report_template.html"), encoding="utf-8").read()

if __name__ == "__main__":
    print(build(sys.argv[1]))
