"""trend_er_split.py の結果（MAの並びだけの全取引 + エントリー時のER）から、ERの効果の検定レポート report.html を作る。
使い方: python trend_results/make_split_report.py <実行フォルダ>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

d = Path(sys.argv[1])
t = pd.read_csv(d / "trades_ma_only.csv"); t["EntryTime"] = pd.to_datetime(t.EntryTime, utc=True)
t["Y"] = t.EntryTime.dt.tz_convert("America/New_York").dt.year
t["cl"] = t.Pair + "_" + t.EntryTime.dt.strftime("%Y%m")
PER = (10, 20, 30); THS = (0.2, 0.3, 0.4)
rng = np.random.default_rng(0)
cls = lambda x: "pos" if x > 0.0005 else "neg" if x < -0.0005 else ""

def boot(mask, B=4000):
    """ペア×月のまとまりごとに取引を引き直し、(高群の平均R − 低群の平均R) の分布を作る"""
    g = t.assign(h=mask).groupby("cl").apply(lambda x: pd.Series(dict(hs=x.R[x.h].sum(), hn=x.h.sum(), ls=x.R[~x.h].sum(), ln=(~x.h).sum())))
    a = g.values; n = len(a); out = []
    for _ in range(B):
        s = a[rng.integers(0, n, n)].sum(0)
        if s[1] > 0 and s[3] > 0: out.append(s[0] / s[1] - s[2] / s[3])
    o = np.array(out); return np.percentile(o, [2.5, 97.5]), float(np.mean(o <= 0))

res = {}
for per in PER:
    for th in THS:
        m = (t[f"ER{per}"] >= th).values; hi, lo = t.R[m], t.R[~m]
        ci, p = boot(pd.Series(m, index=t.index))
        isb, oob = t.Y <= 2023, t.Y >= 2024
        pp = t.assign(h=m).groupby(["Pair", "h"]).R.mean().unstack()
        res[(per, th)] = dict(nh=int(m.sum()), nl=int((~m).sum()), ah=hi.mean(), al=lo.mean(), diff=hi.mean() - lo.mean(), lo_ci=ci[0], hi_ci=ci[1], p=p,
                              d_is=t.R[m & isb].mean() - t.R[~m & isb].mean(), d_oos=t.R[m & oob].mean() - t.R[~m & oob].mean(), pairs=int((pp[True] > pp[False]).sum()))
rho = {per: t[f"ER{per}"].corr(t.R, method="spearman") for per in PER}

def tbl(head, rows): return '<div class="tw"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
rows = []
for (per, th), r in res.items():
    sig = "ci" if r["lo_ci"] > 0 else ""
    rows.append(f'<tr><td>ER({per}) ≥{th:g}</td><td class="num">{r["nh"]}</td><td class="num {cls(r["ah"])}">{r["ah"]:+.3f}</td><td class="num">{r["nl"]}</td><td class="num {cls(r["al"])}">{r["al"]:+.3f}</td>'
                f'<td class="num {cls(r["diff"])}"><b>{r["diff"]:+.3f}</b></td><td class="num {sig}">{r["lo_ci"]:+.3f}〜{r["hi_ci"]:+.3f}</td><td class="num">{r["p"]:.3f}</td>'
                f'<td class="num {cls(r["d_is"])}">{r["d_is"]:+.3f}</td><td class="num {cls(r["d_oos"])}">{r["d_oos"]:+.3f}</td><td class="num">{r["pairs"]}/15</td></tr>')
main_tbl = tbl(["ER≥しきい値","高群 取引数","高群 平均R","低群 取引数","低群 平均R","差","差の95%区間","片側p","差〜2023","差2024〜","高群が良いペア"], rows)

# 五分位の棒（ER20）
def quint(per):
    q = pd.qcut(t[f"ER{per}"], 5, labels=False)
    return t.groupby(q).agg(n=("R", "size"), lo=(f"ER{per}", "min"), hi=(f"ER{per}", "max"), avg=("R", "mean"))
def bars(per):
    g = quint(per); w, h, l, b, tp = 760, 210, 46, 44, 12
    se = t.R.std() / np.sqrt(g.n.iloc[0])
    top = max(abs(g.avg).max(), se) * 1.25; lo_, hi_ = -top, top
    yy = lambda v: tp + (h - tp - b) * (hi_ - v) / (hi_ - lo_)
    gw = (w - l - 10) / 5; bw = gw * .5
    ticks = [-0.1, -0.05, 0, 0.05, 0.1]; grid = "".join(f'<line x1="{l}" x2="{w-10}" y1="{yy(v):.1f}" y2="{yy(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{l-6}" y="{yy(v)+4:.1f}" class="ax" text-anchor="end">{v:+.2f}</text>' for v in ticks if lo_ <= v <= hi_)
    o = ""
    for i, r in g.iterrows():
        cx = l + gw * (i + .5); v = r.avg
        o += f'<rect x="{cx-bw/2:.1f}" y="{min(yy(v),yy(0)):.1f}" width="{bw:.1f}" height="{max(abs(yy(v)-yy(0)),.8):.1f}" fill="var(--{"pos" if v>0 else "neg"})"/>'
        o += f'<text x="{cx:.1f}" y="{yy(v)+(-6 if v>0 else 14):.1f}" class="ax" text-anchor="middle">{v:+.3f}</text>'
        o += f'<text x="{cx:.1f}" y="{h-24}" class="ax" text-anchor="middle">{r.lo:.2f}〜{r.hi:.2f}</text><text x="{cx:.1f}" y="{h-9}" class="ax" text-anchor="middle">{int(r.n)}回</text>'
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="ER({per})の五分位ごとの平均R"><g>{grid}</g>{o}</svg>', se
b20, se20 = bars(20)
qt = []
for per in PER:
    g = quint(per); qt.append(f'<tr><td>ER({per})</td>' + "".join(f'<td class="num {cls(v)}">{v:+.3f}</td>' for v in g.avg) + f'<td class="num">{rho[per]:+.3f}</td></tr>')
q_tbl = tbl(["", "最低20%", "20〜40%", "40〜60%", "60〜80%", "最高20%", "順位相関"], qt)

# 年別（ER20>=0.3）
m = t.ER20 >= 0.3; yrs = t.assign(h=m).groupby(["Y", "h"]).R.agg(["size", "mean"]).unstack()
yr = "".join(f'<tr><td>{y}</td><td class="num">{int(yrs.loc[y,("size",True)])}</td><td class="num {cls(yrs.loc[y,("mean",True)])}">{yrs.loc[y,("mean",True)]:+.3f}</td><td class="num">{int(yrs.loc[y,("size",False)])}</td><td class="num {cls(yrs.loc[y,("mean",False)])}">{yrs.loc[y,("mean",False)]:+.3f}</td><td class="num {cls(yrs.loc[y,("mean",True)]-yrs.loc[y,("mean",False)])}">{yrs.loc[y,("mean",True)]-yrs.loc[y,("mean",False)]:+.3f}</td></tr>' for y in yrs.index)
yr_tbl = tbl(["年", "高群 取引数", "高群 平均R", "低群 取引数", "低群 平均R", "差"], [yr])
sd = t.assign(h=m).groupby(["Side", "h"]).R.agg(["size", "mean"])
sd_rows = "".join(f'<tr><td>{"買い" if s=="L" else "売り"}</td><td class="num">{int(sd.loc[(s,True),"size"])}</td><td class="num {cls(sd.loc[(s,True),"mean"])}">{sd.loc[(s,True),"mean"]:+.3f}</td><td class="num">{int(sd.loc[(s,False),"size"])}</td><td class="num {cls(sd.loc[(s,False),"mean"])}">{sd.loc[(s,False),"mean"]:+.3f}</td></tr>' for s in ("L", "S"))
sd_tbl = tbl(["", "高群 取引数", "高群 平均R", "低群 取引数", "低群 平均R"], [sd_rows])

r20 = res[(20, 0.3)]; r20b = res[(20, 0.2)]
n_ci = sum(1 for r in res.values() if r["lo_ci"] > 0)
doc = f"""<title>ERフィルターの効果検定</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --pos:#18865a; --neg:#c8413a; --gray:#8a95a3;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.65 var(--f-head); }}
.wrap {{ max-width:860px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:28px; }} .wrap>* {{ min-width:0; }}
h1 {{ font-size:1.45rem; line-height:1.3; margin:0; text-wrap:balance; }} h2 {{ font-size:1.05rem; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; }} .sub,.muted {{ color:var(--muted); }} .sub {{ font-size:.85rem; margin-top:6px; }}
.num {{ font-family:var(--f-num); font-variant-numeric:tabular-nums; text-align:right; }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }} td.ci {{ color:var(--pos); font-weight:600; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.verdict {{ border:1.5px solid var(--accent); background:var(--surface); border-radius:12px; padding:14px; }}
.tw {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; white-space:nowrap; }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-weight:500; font-size:.76rem; text-align:right; }} th:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }}
svg {{ width:100%; height:auto; display:block; }} .gl {{ stroke:var(--line); stroke-width:1; }} .gl.z {{ stroke:var(--gray); }} .ax {{ fill:var(--muted); font:11px var(--f-num); }}
ul {{ margin:0; padding-left:1.2em; }} li {{ margin-bottom:4px; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>ERフィルターの効果検定</h1><p class="sub">日足MAの並びだけで入った全{len(t):,}取引（15ペア、2021〜2025）を、エントリー時の日足ER が高い群と低い群に分けて平均Rを比べる。エントリー条件は固定なので、取引の入れ替わりの影響を受けません。</p></header>

<section class="verdict"><p><b>ER が高い群の方が、9通り（期間3×しきい値3）すべてで平均Rが高い。差は +{min(r['diff'] for r in res.values()):.3f}〜+{max(r['diff'] for r in res.values()):.3f}R / 取引。</b></p>
<p class="muted">ER(20)・0.3 の差は {r20['diff']:+.3f}R（95%区間 {r20['lo_ci']:+.3f}〜{r20['hi_ci']:+.3f}）で、区間の下限は 0 に近いものの、偶然とは言いにくい水準です。9通りのうち区間が0を超えるのは{n_ci}通りです。差は2021〜2023年に集中し、2024〜2025年は小さいか逆向きです。効果は存在しそうですが、小さく、安定していません。</p></section>

<section><h2>高群と低群の比較（1取引あたりの平均R）</h2>{main_tbl}
<p class="sub" style="margin-top:8px">高群は ER が「しきい値以上」。差の95%区間は、ペア×月のまとまりで取引を4,000回引き直して求めました（同じ時期の取引が似た値動きをすることを考慮）。片側pは「差が0以下」になる割合です。9通りを同時に見ているので、p は甘めに出ます（0.05 未満でも、9回の試行で1回は偶然に出る程度）。</p></section>

<section><h2>ER(20) の五分位ごとの平均R</h2><div class="card">{b20}</div>
<p class="sub" style="margin-top:8px">ER の低い順に取引を5等分。ER(20) では、最低20%・2番目が負で、上に行くほど平均Rが上がります。目安として、1区分（約{int(len(t)/5)}取引）の平均Rの誤差は ±{se20:.3f}R 程度です。</p>
{q_tbl}</section>

<section><h2>年別（ER(20) ≥ 0.3）</h2>{yr_tbl}
<p class="sub" style="margin-top:8px">2021〜2023年は高群が上回り、2024年は逆転、2025年は小幅です。後半に効果が消えたのか、ばらつきなのかは、この期間だけでは決められません。</p></section>

<section><h2>買い・売り別（ER(20) ≥ 0.3）</h2>{sd_tbl}
<p class="sub" style="margin-top:8px">売りは、ER が低いときの平均Rが特に悪い（{sd.loc[('S',False),'mean']:+.3f}R）。ER の効果は、損失が出やすい売りのエントリーを避けた分が大きいです。</p></section>

<section class="card"><h2>読むときの注意</h2><ul>
<li>ER が高い群の平均Rは +0.03〜+0.09R で、コストを引いた後の期待値としては薄い値です。ER は「悪いエントリーを減らす」役割が大きく、「良いエントリーを見つける」力は小さいです。</li>
<li>ER が低い群が負になっているので、フィルターで除外される取引の期待値が負という意味で、フィルターに意味はあります。</li>
<li>フィルターを実際に入れた検証（前回までの結果）では、除外された分の代わりに別の取引が入るため、合計Rはこの表の差と同じにはなりません。</li>
<li>期間は2021〜2025年の5年のみ。2024〜2025年の差は逆向きなので、ER の優位が直近まで続いているとは言えません。</li>
<li>スワップは未考慮です。1取引のRの標準偏差は {t.R.std():.2f} です。</li>
<li>コード: <code>trend_er_split.py</code>、結果: <code>trend_results/{d.name}/</code></li></ul></section>
</div>"""
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
