"""trend_entry_window_sweep.py の結果から report.html を作る。使い方: python trend_results/make_window_report.py <実行フォルダ>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1]); srcs = [Path(x) for x in sys.argv[2:]] or [d]   # 出力先, 結合する実行フォルダ（省略時は出力先だけ）
t = pd.concat([pd.read_csv(x / "trades_all.csv") for x in srcs], ignore_index=True)
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
HRS = sorted(t.Hours.unique()); WIN = {h_: sorted(t[t.Hours == h_].Window.unique()) for h_ in HRS}
HN = lambda h_: f"{h_}時間足"
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
        hl = " class=hl" if hours * w == 24 else ""
        rows.append(f'<tr{hl}><td>{w}本</td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{fs(a["R"])}</td><td class="num {cls(a["avg"],.0005)}">{a["avg"]:+.3f}</td><td class="num">{a["win"]:.0f}%</td><td class="num">{a["PF"]:.2f}</td>'
                    f'<td class="num">{a["DD"]:.0f}</td><td class="num {cls(a["IS"])}">{fs(a["IS"],0)}</td><td class="num {cls(a["OOS"])}">{fs(a["OOS"],0)}</td><td class="num">{a["pos"]}/15</td>'
                    f'<td class="num {cls(sl["R"])}">{fs(sl["R"])}</td><td class="num {cls(m_["R"])}">{fs(m_["R"],0)}</td><td class="num {cls(nf["R"])}">{fs(nf["R"],0)}</td></tr>')
    return tbl(["本数", "取引数", "合計R", "1取引R", "勝率", "PF", "最大DD", "〜2023", "2024〜", "プラスのペア", "+傾き 合計R", "MAのみ 合計R", "フィルターなし 合計R"], rows)
# 時間幅（本数 × 足の長さ）で全ての足を重ねる
SP = {h_: {w * h_: S[(h_, w, "spec")] for w in WIN[h_]} for h_ in HRS}
spans = sorted({x for h_ in HRS for x in SP[h_]}); common = sorted(set.intersection(*[set(SP[h_]) for h_ in HRS]))
def within_between(key):
    M = np.array([[SP[h_][x][key] for h_ in HRS] for x in common])        # 行=共通の時間幅、列=足
    within = float(np.mean(np.std(M, axis=1, ddof=1)))                    # 同じ時間幅で、足の違いによるばらつき
    between = float(np.std(M.mean(axis=1), ddof=1))                       # 時間幅の違いによるばらつき（足の平均）
    return within, between
wb_R, wb_a = within_between("R"), within_between("avg")
COLS = ["var(--c1)", "var(--c2)", "var(--c3)", "var(--c4)"]
def span_chart():
    w_, h_, l, b, tp = 760, 280, 46, 34, 14
    allv = [v["R"] for h2 in HRS for v in SP[h2].values()]; lo, hi = min(min(allv), 0), max(allv); stp = 25
    lo = np.floor(lo / stp) * stp; hi = np.ceil(hi / stp) * stp
    xs = lambda hh: l + 22 + (w_ - l - 60) * spans.index(hh) / (len(spans) - 1); yy = lambda v: tp + (h_ - tp - b) * (hi - v) / (hi - lo)
    g = "".join(f'<line x1="{l}" x2="{w_-14}" y1="{yy(v):.1f}" y2="{yy(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{l-6}" y="{yy(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>' for v in np.arange(lo, hi + 1, stp))
    xl = "".join(f'<text x="{xs(x):.1f}" y="{h_-16}" class="ax" text-anchor="middle">{x}時間</text>' for x in spans)
    o = ""
    for i, h2 in enumerate(HRS):
        ks = sorted(SP[h2]); pts = " L".join(f"{xs(k):.1f},{yy(SP[h2][k]['R']):.1f}" for k in ks)
        o += f'<path d="M{pts}" fill="none" stroke="{COLS[i]}" stroke-width="2"/>' + "".join(f'<circle cx="{xs(k):.1f}" cy="{yy(SP[h2][k]["R"]):.1f}" r="3.2" fill="{COLS[i]}"/>' for k in ks)
    key = "".join(f'<span><i style="background:{COLS[i]}"></i>{HN(h2)}スイング</span>' for i, h2 in enumerate(HRS))
    return f'<svg viewBox="0 0 {w_} {h_}" role="img" aria-label="時間幅ごとの合計R"><g>{g}{xl}</g>{o}</svg><div class="key">{key}</div>'
span_rows = "".join(f'<tr><td>{x}時間</td>' + "".join((f'<td class="num">{v["n"]}</td><td class="num {cls(v["R"])}">{fs(v["R"])}</td><td class="num {cls(v["avg"],.0005)}">{v["avg"]:+.3f}</td>' if (v := SP[h2].get(x)) else '<td></td>' * 3) for h2 in HRS) + '</tr>' for x in spans)
span_tbl = tbl(["時間幅"] + sum([[f"{HN(h2)} 取引数", "合計R", "1取引R"] for h2 in HRS], []), [span_rows])
best = {h_: max(WIN[h_], key=lambda w: S[(h_, w, "spec")]["R"]) for h_ in HRS}
best_txt = "、".join(f"{HN(h_)} {best[h_]}本（{best[h_] * h_}時間・{fs(S[(h_, best[h_], 'spec')]['R'],0)}R・{S[(h_, best[h_], 'spec')]['n']}回）" for h_ in HRS)
negs = [(h_, w) for h_ in HRS for w in WIN[h_] if S[(h_, w, "spec")]["R"] < 0]
neg_txt = "、".join(f"{HN(h_)}{w}本（{w*h_}時間・{fs(S[(h_, w, 'spec')]['R'],0)}R）" for h_, w in negs) or "ありません"
peak_span = max(common, key=lambda x: np.mean([SP[h_][x]["R"] for h_ in HRS]))
common_txt = "、".join(f"{x}時間: " + " / ".join(fs(SP[h_][x]["R"], 0) for h_ in HRS) for x in common)
ratio = wb_R[0] / wb_R[1] if wb_R[1] else float("inf")
if ratio < 0.7: head_txt = f"足の長さ（{' / '.join(HN(h_) for h_ in HRS)}）を変えても、時間幅ごとの成績は近い形になりました。"
elif ratio < 1.3: head_txt = "足の長さによる違いと時間幅による違いが同じくらいの大きさで、時間幅だけでは説明しきれません。"
else: head_txt = "同じ時間幅でも足の長さによる違いの方が大きく、時間幅だけでは説明できません。"
pm = np.mean([SP[h_][peak_span]["R"] for h_ in HRS])
head_txt += f" 共通の時間幅の中では {peak_span}時間が最高（足の平均 {pm:+.0f}R）です。"
se = lambda n: 1.25 * n ** .5
n_peak = [SP[h_][peak_span]["n"] for h_ in HRS]

doc = f"""<title>スイング本数の感度</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --pos:#18865a; --neg:#c8413a; --gray:#8a95a3; --hl:#eef3ff; --c1:#2a5bd7; --c2:#d9822b; --c3:#1f9d8f; --c4:#8e5bd1;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; --c1:#79a2ff; --c2:#f0a85a; --c3:#4fd1c0; --c4:#b896f0; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; --c1:#79a2ff; --c2:#f0a85a; --c3:#4fd1c0; --c4:#b896f0; color-scheme:dark; }}
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
<header><h1>スイング本数の感度</h1><p class="sub">エントリーを Swing High/Low の抜け（前後N本で確定、一度抜かれたラインは次のスイング確定まで無効）にして、{" / ".join(f"{HN(h_)}（{WIN[h_][0]}〜{WIN[h_][-1]}本）" for h_ in HRS)}で N を動かした結果。フィルターは日足MA20/75/200の並び + ER(20)≥0.3。SL/TP は4時間足 ATR(14) × 1.0 / 1.5。15ペア、2021〜2025。</p></header>

<section class="verdict"><p><b>{head_txt}</b></p>
<p class="muted">共通の時間幅（{", ".join(f"{x}時間" for x in common)}）で、同じ時間幅の足違いによるばらつきは合計Rで平均 {wb_R[0]:.0f}R、時間幅の違いによるばらつきは {wb_R[1]:.0f}R です。1取引あたりでは {wb_a[0]:.3f}R と {wb_a[1]:.3f}R です。最良は {best_txt}。マイナスになったのは {neg_txt} です。</p></section>

<section><h2>時間幅で重ねた比較（ER(20)≥0.3の合計R）</h2><div class="card">{span_chart()}</div>
<div style="height:10px"></div>{span_tbl}
<p class="sub" style="margin-top:8px">時間幅は「前後N本 × 足の長さ」です（1時間足24本 = 2時間足12本 = 3時間足8本 = 4時間足6本 = 24時間）。共通の時間幅の合計R（{" / ".join(HN(h_) for h_ in HRS)}の順）: {common_txt}。</p></section>

{"".join(f"""<section><h2>{HN(h_)}スイング: 本数ごとの合計R（ER(20)≥0.3）</h2><div class="card">{bars(h_, WIN[h_], title=HN(h_) + " 合計R")}</div>
<div style="height:10px"></div>{table(h_, WIN[h_])}</section>""" for h_ in HRS)}

<section class="card"><h2>読むときの注意</h2><ul>
<li>誤差の目安: 1取引のRの標準偏差は約1.25R。合計Rの誤差は、取引{min(n_peak)}回で±{se(min(n_peak)):.0f}R、{max(n_peak)}回で±{se(max(n_peak)):.0f}R 程度です（時間幅 {peak_span}時間）。隣り合う本数の取引は大部分が重なっているので、本数が違う結果は独立ではありません。</li>
<li>これまでに試した方式・本数は延べ{len(t.groupby(['Hours','Window']))}通り以上あります。最良の本数だけを選ぶと合わせ込みになります。</li>
<li>足の長さが違っても、同じ時間幅のスイングは、ほぼ同じ価格の山谷から作られます。そのため、足ごとの結果が近いのは、独立した確認というより、足の長さが成績にほとんど影響しないことを示しています。実質的に動かしているパラメータは「時間幅」の1つです。</li>
<li>「+傾き」は、さらに20MAの傾き（1日）を足した場合の合計R、「MAのみ」は ER を使わない場合の合計Rです。</li>
<li>スワップは未考慮です。データは2021〜2025年の5年間のみです。</li>
<li>コード: <code>trend_entry_window_sweep.py</code>、結果: <code>trend_results/{d.name}/</code></li></ul></section>
</div>"""
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
