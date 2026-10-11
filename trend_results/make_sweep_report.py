"""trend_threshold_sweep.py の結果から report.html を作る。使い方: python trend_results/make_sweep_report.py <実行フォルダ>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

d = Path(sys.argv[1])
t = pd.read_csv(d / "trades_all.csv"); on = pd.read_csv(d / "filter_on_share.csv")
t["ExitTime"] = pd.to_datetime(t.ExitTime, utc=True); t["EntryTime"] = pd.to_datetime(t.EntryTime, utc=True)
t["Y"] = t.EntryTime.dt.tz_convert("America/New_York").dt.year
onm = on.groupby(["Period", "Threshold"]).OnPct.mean()
PER = sorted(t.Period.unique()); TH = sorted(t.Threshold.unique())

def st(g):
    r = g.sort_values("ExitTime").R.values; eq = np.cumsum(r); gl = -r[r < 0].sum()
    return dict(n=len(r), R=r.sum(), avg=r.mean() if len(r) else 0, PF=r[r > 0].sum() / gl if gl > 0 else float("nan"),
                DD=float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(r) else 0,
                IS=g[g.Y <= 2023].R.sum(), OOS=g[g.Y >= 2024].R.sum(), pos=int((g.groupby("Pair").R.sum() > 0).sum()))
S = {(p, h): st(g) for (p, h), g in t.groupby(["Period", "Threshold"])}
cls = lambda x: "pos" if x > 0.05 else "neg" if x < -0.05 else ""
fs = lambda x, p=1: f"{x:+.{p}f}"

# 折れ線: 合計R × しきい値
COL = {PER[0]: "var(--c1)", PER[1]: "var(--c2)", PER[2]: "var(--c3)"}
W, H, L, B, T = 760, 300, 46, 34, 14
allr = [S[k]["R"] for k in S]; lo, hi = min(min(allr), 0), max(allr)
step = 25; lo = np.floor(lo / step) * step; hi = np.ceil(hi / step) * step
x = lambda h: L + (W - L - 70) * (TH.index(h)) / (len(TH) - 1)
y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
grid = "".join(f'<line x1="{L}" x2="{W-70}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{L-6}" y="{y(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>' for v in np.arange(lo, hi + 1, step))
xl = "".join(f'<text x="{x(h):.1f}" y="{H-14}" class="ax" text-anchor="middle">{h:g}</text>' for h in TH)
lines = ""
for p in PER:
    pts = " L".join(f"{x(h):.1f},{y(S[(p,h)]['R']):.1f}" for h in TH)
    dots = "".join(f'<circle cx="{x(h):.1f}" cy="{y(S[(p,h)]["R"]):.1f}" r="2.8" fill="{COL[p]}"/>' for h in TH)
    last = S[(p, TH[-1])]["R"]
    lines += f'<path d="M{pts}" fill="none" stroke="{COL[p]}" stroke-width="2"/>{dots}<text x="{W-64}" y="{y(last)+4:.1f}" class="lb" fill="{COL[p]}">ER({p})</text>'
svg1 = f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="しきい値ごとの合計R"><g>{grid}{xl}</g><text x="{(L+W-70)/2:.0f}" y="{H-1}" class="ax" text-anchor="middle">ER しきい値（0 = MAの並びだけ）</text>{lines}</svg>'

# 棒: 前半・後半（期間ごと）
def bars(p):
    w, h, l, b, tp = 760, 190, 46, 30, 10
    vals = [S[(p, k)][s] for k in TH for s in ("IS", "OOS")]
    lo_, hi_ = min(min(vals), 0), max(vals); stp = 25
    lo_ = np.floor(lo_ / stp) * stp; hi_ = np.ceil(hi_ / stp) * stp
    yy = lambda v: tp + (h - tp - b) * (hi_ - v) / (hi_ - lo_)
    gw = (w - l - 10) / len(TH); bw = gw * .32
    g = "".join(f'<line x1="{l}" x2="{w-10}" y1="{yy(v):.1f}" y2="{yy(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{l-6}" y="{yy(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>' for v in np.arange(lo_, hi_ + 1, stp))
    out = ""
    for i, k in enumerate(TH):
        cx = l + gw * (i + .5)
        for j, (s, c) in enumerate((("IS", "var(--c1)"), ("OOS", "var(--c2)"))):
            v = S[(p, k)][s]; x0 = cx + (j - 1) * bw - (0 if j else 0) + (bw * .0)
            x0 = cx - bw - 1 if j == 0 else cx + 1
            out += f'<rect x="{x0:.1f}" y="{min(yy(v), yy(0)):.1f}" width="{bw:.1f}" height="{max(abs(yy(v)-yy(0)),.8):.1f}" fill="{c}"/>'
        out += f'<text x="{cx:.1f}" y="{h-12}" class="ax" text-anchor="middle">{k:g}</text>'
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="ER({p}) 前半と後半の合計R"><g>{g}</g>{out}</svg>'

def table(p):
    rows = []
    for h in TH:
        s = S[(p, h)]
        rows.append(f'<tr><td class="num">{h:g}</td><td class="num">{onm[(p,h)]:.0f}%</td><td class="num">{s["n"]}</td><td class="num {cls(s["R"])}">{fs(s["R"])}</td>'
                    f'<td class="num">{s["avg"]:+.3f}</td><td class="num">{s["PF"]:.2f}</td><td class="num">{s["DD"]:.0f}</td>'
                    f'<td class="num {cls(s["IS"])}">{fs(s["IS"],0)}</td><td class="num {cls(s["OOS"])}">{fs(s["OOS"],0)}</td><td class="num">{s["pos"]}/15</td></tr>')
    head = "".join(f"<th>{c}</th>" for c in ["しきい値", "ON期間", "取引数", "合計R", "1取引R", "PF", "最大DD", "〜2023", "2024〜", "プラスのペア"])
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

base = S[(PER[0], 0.0)]
best = {p: max(TH, key=lambda h: S[(p, h)]["R"]) for p in PER}
p20 = 20 if 20 in PER else PER[1]
plateau = [h for h in TH if S[(p20, h)]["R"] > 50]
oos_pos = sum(1 for k in S if S[k]["OOS"] > 0 and k[1] > 0)
oos_tot = sum(1 for k in S if k[1] > 0)

doc = f"""<title>ERしきい値の感度</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7;
  --pos:#18865a; --neg:#c8413a; --gray:#8a95a3; --c1:#2a5bd7; --c2:#d9822b; --c3:#1f9d8f;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae;
  --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --c1:#79a2ff; --c2:#f0a85a; --c3:#4fd1c0; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae;
  --accent:#79a2ff; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --c1:#79a2ff; --c2:#f0a85a; --c3:#4fd1c0; color-scheme:dark; }}
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
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-weight:500; font-size:.76rem; }}
th {{ text-align:right; }} tr:last-child td {{ border-bottom:0; }}
svg {{ width:100%; height:auto; display:block; }} .gl {{ stroke:var(--line); stroke-width:1; }} .gl.z {{ stroke:var(--gray); }}
.ax {{ fill:var(--muted); font:11px var(--f-num); }} .lb {{ font:600 12px var(--f-head); }}
.key {{ display:flex; gap:14px; flex-wrap:wrap; font-size:.82rem; color:var(--muted); margin-top:6px; }} .key i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; }}
ul {{ margin:0; padding-left:1.2em; }} li {{ margin-bottom:4px; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>ERしきい値の感度</h1><p class="sub">日足MA20/75/200の並び + ER でロング/ショートを許可。ER期間 {PER[0]}・{PER[1]}・{PER[2]} × しきい値 {TH[0]:g}〜{TH[-1]:g}。15ペア、2021〜2025、4H 18本ブレイク SL1.0/TP1.5 ATR。</p></header>

<section class="verdict"><p><b>ER を入れると、MAの並びだけ（{fs(base['R'])}R）より大きく改善します。最良は ER({p20}) のしきい値 {best[p20]:g} で {fs(S[(p20, best[p20])]['R'],0)}R。</b></p>
<p class="muted">ER({p20}) は しきい値 {TH[1]:g}〜{TH[-3]:g} の範囲でプラスが続き、0.2〜0.4 で +54〜+112R です。しきい値 0.45 で急に弱くなり、のこぎり状に動くため、細かい最良値に意味はありません。後半（2024〜）が改善する設定は、ER を使った {oos_tot} 通りのうち {oos_pos} 通りだけです。</p></section>

<section><h2>合計R（15ペア合算）</h2><div class="card">{svg1}</div>
<p class="sub">しきい値 0 は MA の並びだけ。ER({PER[0]})・ER({PER[1]})・ER({PER[2]}) は同じ線の上に出発して、しきい値を上げるほど取引数が減ります。</p></section>

{''.join(f"""<section><h2>ER({p}) の内訳</h2><div class="card">{bars(p)}<div class="key"><span><i style="background:var(--c1)"></i>〜2023年（前半）</span><span><i style="background:var(--c2)"></i>2024〜2025年（後半）</span></div></div>
<div style="height:10px"></div>{table(p)}</section>""" for p in PER)}

<section class="card"><h2>読むときの注意</h2><ul>
<li>売買ルールは添付ルールのまま（ブレイク18本、SL 1.0 / TP 1.5 ATR）で、ER のフィルターだけを変えています。</li>
<li>全期間を見て並べた表です。最良のしきい値を選ぶと合わせ込みになります。しきい値を決めるなら、成績が高い一点ではなく、周辺も安定している範囲を使うのが安全です。</li>
<li>しきい値が高いほど取引数が少なく（ER(30)・0.6 は45回）、PF が高くても偶然の幅が大きくなります。</li>
<li>前半・後半は取引のエントリー年で分けた集計です。ER を使った設定は、どれも後半が前半より弱くなっています（MA の並びだけは後半の方が良い）。</li>
<li>スワップは未考慮です。ER({p20}) のしきい値 0.3 の行は、前回のレポートと取引数・合計Rが一致します。</li>
<li>コード: <code>trend_threshold_sweep.py</code>、結果: <code>trend_results/{d.name}/</code></li></ul></section>
</div>"""
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
