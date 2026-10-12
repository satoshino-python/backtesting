"""trend_entry_window_sweep.py の結果から report.html を作る。使い方: python trend_results/make_window_report.py <実行フォルダ>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1])
t = pd.read_csv(d / "trades_all.csv")
t["ExitTime"] = pd.to_datetime(t.ExitTime, utc=True); t["EntryTime"] = pd.to_datetime(t.EntryTime, utc=True)
t["Y"] = t.EntryTime.dt.tz_convert("America/New_York").dt.year
cls = lambda x, e=0.05: "pos" if x > e else "neg" if x < -e else ""
fs = lambda x, p=1: f"{x:+.{p}f}"
def st(g):
    g = g.sort_values("ExitTime"); r = g.R.values; eq = np.cumsum(r); gl = -r[r < 0].sum()
    return dict(n=len(r), R=r.sum(), avg=r.mean() if len(r) else 0, win=(r > 0).mean() * 100 if len(r) else 0, PF=r[r > 0].sum() / gl if gl > 0 else float("nan"),
                DD=float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(r) else 0, IS=g[g.Y <= 2023].R.sum(), OOS=g[g.Y >= 2024].R.sum(),
                pos=int((g.groupby("Pair").R.sum() > 0).sum()))
S = {k: st(g) for k, g in t.groupby(["Hours", "Window", "Variant"])}
H1 = sorted(t[t.Hours == 1].Window.unique()); H4 = sorted(t[t.Hours == 4].Window.unique())
def tbl(head, rows): return '<div class="tw"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
def bars(hours, wins, key="R", variant="spec", title="", fmt=lambda v: f"{v:+.0f}"):
    w_, h_, l, b, tp = 760, 200, 46, 30, 14
    vals = [S[(hours, w, variant)][key] for w in wins]; lo, hi = min(min(vals), 0), max(max(vals), 0)
    pad = (hi - lo) * .12 or 1; lo -= pad if lo < 0 else 0; hi += pad
    yy = lambda v: tp + (h_ - tp - b) * (hi - v) / (hi - lo); gw = (w_ - l - 10) / len(wins); bw = gw * .5
    ticks = np.linspace(lo, hi, 5)
    grid = f'<line x1="{l}" x2="{w_-10}" y1="{yy(0):.1f}" y2="{yy(0):.1f}" class="gl z"/>' + "".join(f'<text x="{l-6}" y="{yy(v)+4:.1f}" class="ax" text-anchor="end">{fmt(v)}</text>' for v in ticks)
    o = ""
    for i, (w, v) in enumerate(zip(wins, vals)):
        cx = l + gw * (i + .5)
        o += f'<rect x="{cx-bw/2:.1f}" y="{min(yy(v), yy(0)):.1f}" width="{bw:.1f}" height="{max(abs(yy(v)-yy(0)),.8):.1f}" fill="var(--{"pos" if v>0 else "neg"})"/>'
        o += f'<text x="{cx:.1f}" y="{yy(v)+(-5 if v>=0 else 13):.1f}" class="ax" text-anchor="middle">{fmt(v)}</text><text x="{cx:.1f}" y="{h_-10}" class="ax" text-anchor="middle">{w}本</text>'
    return f'<svg viewBox="0 0 {w_} {h_}" role="img" aria-label="{title}"><g>{grid}</g>{o}</svg>'
def table(hours, wins):
    rows = []
    for w in wins:
        a, m_, sl, nf = S[(hours, w, "spec")], S[(hours, w, "ma_only")], S[(hours, w, "spec_slope1")], S[(hours, w, "nofilter")]
        hl = " class=hl" if (hours, w) in ((1, 24), (4, 18)) else ""
        rows.append(f'<tr{hl}><td>{w}本</td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{fs(a["R"])}</td><td class="num {cls(a["avg"],.0005)}">{a["avg"]:+.3f}</td><td class="num">{a["win"]:.0f}%</td><td class="num">{a["PF"]:.2f}</td>'
                    f'<td class="num">{a["DD"]:.0f}</td><td class="num {cls(a["IS"])}">{fs(a["IS"],0)}</td><td class="num {cls(a["OOS"])}">{fs(a["OOS"],0)}</td><td class="num">{a["pos"]}/15</td>'
                    f'<td class="num {cls(sl["R"])}">{fs(sl["R"])}</td><td class="num {cls(m_["R"])}">{fs(m_["R"],0)}</td><td class="num {cls(nf["R"])}">{fs(nf["R"],0)}</td></tr>')
    return tbl(["本数", "取引数", "合計R", "1取引R", "勝率", "PF", "最大DD", "〜2023", "2024〜", "プラスのペア", "+傾き 合計R", "MAのみ 合計R", "フィルターなし 合計R"], rows)
def spread(hours, wins):
    r = [S[(hours, w, "spec")]["R"] for w in wins]; return min(r), max(r), sum(1 for x in r if x > 0), len(r)
a1, a4 = spread(1, H1), spread(4, H4)
best1 = max(H1, key=lambda w: S[(1, w, "spec")]["R"]); best4 = max(H4, key=lambda w: S[(4, w, "spec")]["R"])
# 滑らかさ: 隣り合う本数の合計Rの差の平均 / 全体の幅
sm = lambda hours, wins: np.mean([abs(S[(hours, wins[i+1], "spec")]["R"] - S[(hours, wins[i], "spec")]["R"]) for i in range(len(wins) - 1)])
# 取引数と合計Rの関係
nR = lambda hours, wins: np.corrcoef([S[(hours, w, "spec")]["n"] for w in wins], [S[(hours, w, "spec")]["R"] for w in wins])[0, 1]
r4_18 = S[(4, 18, "spec")]; r1_24 = S[(1, 24, "spec")]
se = lambda n: 1.25 * n ** .5

# 時間幅（本数 × 足の長さ）で1時間足と4時間足を重ねる
span1 = {w * 1: S[(1, w, "spec")] for w in H1}; span4 = {w * 4: S[(4, w, "spec")] for w in H4}
spans = sorted(set(span1) | set(span4)); common = sorted(set(span1) & set(span4))
corr = np.corrcoef([span1[x]["R"] for x in common], [span4[x]["R"] for x in common])[0, 1] if len(common) > 2 else float("nan")
corr_a = np.corrcoef([span1[x]["avg"] for x in common], [span4[x]["avg"] for x in common])[0, 1] if len(common) > 2 else float("nan")
def span_chart():
    w_, h_, l, b, tp = 760, 260, 46, 34, 14
    allv = [v["R"] for v in list(span1.values()) + list(span4.values())]; lo, hi = min(min(allv), 0), max(allv); stp = 25
    lo = np.floor(lo / stp) * stp; hi = np.ceil(hi / stp) * stp
    xs = lambda hh: l + 22 + (w_ - l - 60) * spans.index(hh) / (len(spans) - 1); yy = lambda v: tp + (h_ - tp - b) * (hi - v) / (hi - lo)
    g = "".join(f'<line x1="{l}" x2="{w_-14}" y1="{yy(v):.1f}" y2="{yy(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{l-6}" y="{yy(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>' for v in np.arange(lo, hi + 1, stp))
    xl = "".join(f'<text x="{xs(x):.1f}" y="{h_-16}" class="ax" text-anchor="middle">{x}時間</text>' for x in spans)
    o = ""
    for dct, col, nm, dy in ((span1, "var(--c1)", "1時間足", -9), (span4, "var(--c2)", "4時間足", 15)):
        ks = sorted(dct); pts = " L".join(f"{xs(k):.1f},{yy(dct[k]['R']):.1f}" for k in ks)
        o += f'<path d="M{pts}" fill="none" stroke="{col}" stroke-width="2"/>' + "".join(f'<circle cx="{xs(k):.1f}" cy="{yy(dct[k]["R"]):.1f}" r="3.2" fill="{col}"/><text x="{xs(k):.1f}" y="{yy(dct[k]["R"])+dy:.1f}" class="ax" text-anchor="middle" fill="{col}">{dct[k]["R"]:+.0f}</text>' for k in ks)
    key = '<span><i style="background:var(--c1)"></i>1時間足スイング</span><span><i style="background:var(--c2)"></i>4時間足スイング</span>'
    return f'<svg viewBox="0 0 {w_} {h_}" role="img" aria-label="時間幅ごとの合計R"><g>{g}{xl}</g>{o}</svg><div class="key">{key}</div>'
span_rows = "".join(f'<tr><td>{x}時間</td>' + "".join((f'<td class="num">{v["n"]}</td><td class="num {cls(v["R"])}">{fs(v["R"])}</td><td class="num {cls(v["avg"],.0005)}">{v["avg"]:+.3f}</td>' if (v := dct.get(x)) else '<td></td>' * 3) for dct in (span1, span4)) + '</tr>' for x in spans)
negs = [(h_, w) for h_, ws in ((1, H1), (4, H4)) for w in ws if S[(h_, w, "spec")]["R"] < 0]
neg_txt = "、".join(f"{'1時間足' if h_ == 1 else '4時間足'}{w}本（{w*h_}時間・{fs(S[(h_, w, 'spec')]['R'],0)}R・{S[(h_, w, 'spec')]['n']}回）" for h_, w in negs) or "ありません"
span_tbl = tbl(["時間幅", "1H 取引数", "合計R", "1取引R", "4H 取引数", "合計R", "1取引R"], [span_rows])
doc = f"""<title>スイング本数の感度</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --pos:#18865a; --neg:#c8413a; --gray:#8a95a3; --hl:#eef3ff; --c1:#2a5bd7; --c2:#d9822b;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; --c1:#79a2ff; --c2:#f0a85a; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; --c1:#79a2ff; --c2:#f0a85a; color-scheme:dark; }}
.key {{ display:flex; gap:14px; flex-wrap:wrap; font-size:.82rem; color:var(--muted); margin-top:6px; }} .key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.65 var(--f-head); }}
.wrap {{ max-width:860px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:28px; }} .wrap>* {{ min-width:0; }}
h1 {{ font-size:1.45rem; line-height:1.3; margin:0; text-wrap:balance; }} h2 {{ font-size:1.05rem; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; }} .sub,.muted {{ color:var(--muted); }} .sub {{ font-size:.85rem; margin-top:6px; }}
.num {{ font-family:var(--f-num); font-variant-numeric:tabular-nums; text-align:right; }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.verdict {{ border:1.5px solid var(--accent); background:var(--surface); border-radius:12px; padding:14px; }}
.tw {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; white-space:nowrap; }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-weight:500; font-size:.76rem; text-align:right; }} th:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }} tr.hl td {{ background:var(--hl); }}
svg {{ width:100%; height:auto; display:block; }} .gl {{ stroke:var(--line); stroke-width:1; }} .gl.z {{ stroke:var(--gray); }} .ax {{ fill:var(--muted); font:11px var(--f-num); }}
ul {{ margin:0; padding-left:1.2em; }} li {{ margin-bottom:4px; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>スイング本数の感度</h1><p class="sub">エントリーを Swing High/Low の抜け（前後N本で確定、一度抜かれたラインは次のスイング確定まで無効）にして、1時間足（6〜48本）と4時間足（3〜18本）で N を動かした結果。フィルターは日足MA20/75/200の並び + ER(20)≥0.3。SL/TP は4時間足 ATR(14) × 1.0 / 1.5。15ペア、2021〜2025。</p></header>

<section class="verdict"><p><b>成績は足の長さより、スイングの時間幅（本数×足の長さ）に左右されているように見えます。1時間足と4時間足で、同じ時間幅の合計Rが近い値になりました（相関 {corr:+.2f}、ただし共通の時間幅は {len(common)}点だけ）。</b></p><p><b>1時間足は本数 {H1[0]}〜{H1[-1]} の合計Rが {fs(a1[0],0)}〜{fs(a1[1],0)}R（{a1[2]}/{a1[3]}通りがプラス）、4時間足は本数 {H4[0]}〜{H4[-1]} で {fs(a4[0],0)}〜{fs(a4[1],0)}R（{a4[2]}/{a4[3]}通りがプラス）。</b></p>
<p class="muted">最良は 1時間足 {best1}本（{fs(S[(1,best1,'spec')]['R'],0)}R・{S[(1,best1,'spec')]['n']}回）、4時間足 {best4}本（{fs(S[(4,best4,'spec')]['R'],0)}R・{S[(4,best4,'spec')]['n']}回）。マイナスになったのは {neg_txt} です。</p></section>

<section><h2>時間幅で重ねた比較（ER(20)≥0.3の合計R）</h2><div class="card">{span_chart()}</div>
<div style="height:10px"></div>{span_tbl}
<p class="sub" style="margin-top:8px">時間幅は「前後N本 × 足の長さ」です（1時間足24本 = 4時間足6本 = 24時間）。共通の時間幅 {", ".join(f"{x}時間" for x in common)} で、1時間足と4時間足の合計Rは {", ".join(f"{fs(span1[x]['R'],0)} / {fs(span4[x]['R'],0)}" for x in common)}R です。</p></section>

<section><h2>1時間足スイング: 本数ごとの合計R（ER(20)≥0.3）</h2><div class="card">{bars(1, H1, title="1時間足 合計R")}</div>
<div style="height:10px"></div>{table(1, H1)}</section>

<section><h2>4時間足スイング: 本数ごとの合計R（ER(20)≥0.3）</h2><div class="card">{bars(4, H4, title="4時間足 合計R")}</div>
<div style="height:10px"></div>{table(4, H4)}
<p class="sub" style="margin-top:8px">色付きの行は、これまでのレポートの設定（1時間足24本、4時間足18本）です。</p></section>

<section><h2>1取引あたりの平均R</h2><div class="card">{bars(1, H1, key="avg", title="1時間足 平均R", fmt=lambda v: f"{v:+.2f}")}</div>
<div style="height:10px"></div><div class="card">{bars(4, H4, key="avg", title="4時間足 平均R", fmt=lambda v: f"{v:+.2f}")}</div></section>

<section class="card"><h2>読むときの注意</h2><ul>
<li>誤差の目安: 1取引のRの標準偏差は約1.25R。合計Rの誤差は、取引{r4_18['n']}回で±{se(r4_18['n']):.0f}R、{r1_24['n']}回で±{se(r1_24['n']):.0f}R 程度です。隣り合う本数の取引は大部分が重なっているので、本数が違う結果は独立ではありません。</li>
<li>これまでの2つの方式（4時間足18本・1時間足24本）に加えて、今回は11通りを試しています。最良の本数だけを選ぶと合わせ込みになります。</li>
<li>「+傾き」は、さらに20MAの傾き（1日）を足した場合の合計R、「MAのみ」は ER を使わない場合の合計Rです。</li>
<li>スワップは未考慮です。データは2021〜2025年の5年間のみです。</li>
<li>コード: <code>trend_entry_window_sweep.py</code>、結果: <code>trend_results/{d.name}/</code></li></ul></section>
</div>"""
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
