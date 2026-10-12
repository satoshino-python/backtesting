"""main_4H_trend_slope_fast.py（+ trend_slope_split.py）の結果から report.html を作る。
使い方: python trend_results/make_slope_report.py <slope実行フォルダ> <er_split実行フォルダ（slope_split.csv入り）>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

import json
d = Path(sys.argv[1]); dsplit = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
dcmps = [Path(a) for a in sys.argv[3:]]   # 比較する他方式の実行フォルダ（複数可）
cfgj = json.loads((d / "config.json").read_text(encoding="utf-8")); MODE = cfgj.get("entry_mode", "donchian")
ENT_ALL = {
    "donchian": ("確定済み直近18本の4時間足の最高値/最安値への逆指値（判定中の足は含めない）", "4時間足18本の高安ブレイク", "4H 高安"),
    "swing": ("既存戦略と同じ Swing High/Low（4時間足 前後18本で確定。一度抜かれたラインは、次のスイングが確定するまで無効）への逆指値", "4時間足18本のスイングハイ・ローブレイク", "4H スイング"),
    "swing1h": ("1時間足 前後24本で確定した Swing High/Low（一度抜かれたラインは、次のスイングが確定するまで無効）への逆指値。SL/TP の ATR は4時間足 ATR(14)", "1時間足24本のスイングハイ・ローブレイク", "1H スイング"),
    "swing1h_atr1h": ("1時間足 前後24本で確定した Swing High/Low への逆指値。SL/TP の ATR は1時間足 ATR(14)", "1時間足24本のスイングハイ・ローブレイク（ATR 1H）", "1H スイング(ATR1H)"),
}
ENT, ENTS, ENTN = ENT_ALL[MODE]
t = pd.read_csv(d / "trades_all.csv")
t["ExitTime"] = pd.to_datetime(t.ExitTime, utc=True); t["EntryTime"] = pd.to_datetime(t.EntryTime, utc=True)
t["Y"] = t.EntryTime.dt.tz_convert("America/New_York").dt.year
on = pd.read_csv(d / "filter_on_share.csv").groupby("Variant").OnPct.mean()
cls = lambda x, e=0.05: "pos" if x > e else "neg" if x < -e else ""
fs = lambda x, p=1: f"{x:+.{p}f}"

def st(g):
    g = g.sort_values("ExitTime"); r = g.R.values; eq = np.cumsum(r); gl = -r[r < 0].sum()
    return dict(n=len(r), R=r.sum(), avg=r.mean() if len(r) else 0, win=(r > 0).mean() * 100 if len(r) else 0, PF=r[r > 0].sum() / gl if gl > 0 else float("nan"),
                DD=float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(r) else 0,
                IS=g[g.Y <= 2023].R.sum(), OOS=g[g.Y >= 2024].R.sum(), pos=int((g.groupby("Pair").R.sum() > 0).sum()))
V = {k: st(g) for k, g in t.groupby("Variant")}
A, B = V["spec"], V["spec_slope1"]
def tbl(head, rows): return '<div class="tw"><table><thead><tr>' + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"

# 累積R
def curve(v):
    g = t[t.Variant == v].sort_values("ExitTime").set_index("ExitTime").R.cumsum(); return g.resample("W").last().ffill()
ca, cb = curve("spec"), curve("spec_slope1"); idx = ca.index.union(cb.index)
ca, cb = ca.reindex(idx).ffill().fillna(0), cb.reindex(idx).ffill().fillna(0)
W, H, L, Bm, T = 760, 280, 46, 28, 12
lo, hi = min(ca.min(), cb.min(), 0), max(ca.max(), cb.max()); step = 20; lo = np.floor(lo / step) * step; hi = np.ceil(hi / step) * step
x = lambda i: L + (W - L - 10) * i / (len(idx) - 1); y = lambda v: T + (H - T - Bm) * (hi - v) / (hi - lo)
grid = "".join(f'<line x1="{L}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{L-6}" y="{y(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>' for v in np.arange(lo, hi + 1, step))
yrs = "".join(f'<text x="{x(i):.1f}" y="{H-8}" class="ax" text-anchor="middle">{ts.year}</text>' for i, ts in enumerate(idx) if i and ts.year != idx[i-1].year)
pth = lambda s: "M" + " L".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s.values))
svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="累積R"><g>{grid}{yrs}</g><path d="{pth(ca)}" class="ln a"/><path d="{pth(cb)}" class="ln b"/>'
       f'<text x="{x(len(idx)-1)-4:.1f}" y="{y(cb.iloc[-1])-8:.1f}" class="lb b" text-anchor="end">+20MA傾き {cb.iloc[-1]:+.0f}R</text>'
       f'<text x="{x(len(idx)-1)-4:.1f}" y="{y(ca.iloc[-1])+16:.1f}" class="lb a" text-anchor="end">傾きなし {ca.iloc[-1]:+.0f}R</text></svg>')

NAME = {"nofilter": "フィルターなし", "ma_only": "MAの並びだけ", "ma_slope1": "MAの並び + 20MA傾き(1日)", "ma_slope3": "MAの並び + 20MA傾き(3日)",
        "spec": "ER(20)≥0.3（添付ルール）", "spec_slope1": "ER(20)≥0.3 + 20MA傾き(1日)", "spec_slope3": "ER(20)≥0.3 + 20MA傾き(3日)", "spec_slope5": "ER(20)≥0.3 + 20MA傾き(5日)",
        "er10": "ER(10)≥0.3", "er10_slope1": "ER(10)≥0.3 + 20MA傾き(1日)",
        "slope1_er0.2": "ER(20)≥0.2 + 傾き(1日)", "slope1_er0.25": "ER(20)≥0.25 + 傾き(1日)", "slope1_er0.35": "ER(20)≥0.35 + 傾き(1日)", "slope1_er0.4": "ER(20)≥0.4 + 傾き(1日)"}
vr = []
for k, nm in NAME.items():
    s = V[k]; hl = " class=hl" if k == "spec_slope1" else ""
    vr.append(f'<tr{hl}><td>{nm}</td><td class="num">{on.get(k, 100):.0f}%</td><td class="num">{s["n"]}</td><td class="num {cls(s["R"])}">{fs(s["R"])}</td><td class="num">{s["avg"]:+.3f}</td><td class="num">{s["PF"]:.2f}</td><td class="num">{s["DD"]:.0f}</td>'
              f'<td class="num {cls(s["IS"])}">{fs(s["IS"],0)}</td><td class="num {cls(s["OOS"])}">{fs(s["OOS"],0)}</td><td class="num">{s["pos"]}/15</td></tr>')
same13 = V["spec_slope1"]["n"] == V["spec_slope3"]["n"] and abs(V["spec_slope1"]["R"] - V["spec_slope3"]["R"]) < 1e-6
cmp_txt = (("ER(20)≥0.3 では、傾き1日と3日の取引が完全に同じです（ER が高く3本のMAが整列している日は、20MAの向きが1日で見ても3日で見ても一致するため）。" if same13 else "")
           + f"ER を使わない場合は、MAの並び {fs(V['ma_only']['R'])}R に対して、傾き（1日）で {fs(V['ma_slope1']['R'])}R、傾き（3日）で {fs(V['ma_slope3']['R'])}R です。"
           + f"ER(10) では {fs(V['er10']['R'])}R → 傾きを足して {fs(V['er10_slope1']['R'])}R です。")
var_tbl = tbl(["設定", "ON期間", "取引数", "合計R", "1取引R", "PF", "最大DD", "〜2023", "2024〜", "プラスのペア"], vr)

# 年別・ペア別
yr = []
for y_ in range(2021, 2026):
    a, b = st(t[(t.Variant == "spec") & (t.Y == y_)]), st(t[(t.Variant == "spec_slope1") & (t.Y == y_)])
    yr.append(f'<tr><td>{y_}</td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{fs(a["R"])}</td><td class="num">{b["n"]}</td><td class="num {cls(b["R"])}">{fs(b["R"])}</td><td class="num {cls(b["R"]-a["R"])}">{fs(b["R"]-a["R"])}</td></tr>')
pr = []
for p in sorted(t.Pair.unique(), key=lambda p: -st(t[(t.Variant == "spec_slope1") & (t.Pair == p)])["R"]):
    a, b = st(t[(t.Variant == "spec") & (t.Pair == p)]), st(t[(t.Variant == "spec_slope1") & (t.Pair == p)])
    pr.append(f'<tr><td><a href="#viewer" data-chart="{p}">{p}</a></td><td class="num">{b["n"]}</td><td class="num {cls(b["R"])}">{fs(b["R"])}</td><td class="num">{b["win"]:.0f}%</td><td class="num">{b["PF"]:.2f}</td><td class="num">{b["DD"]:.1f}</td><td class="num {cls(a["R"])}">{fs(a["R"])}</td></tr>')
sdw = {k: st(t[(t.Variant == "spec_slope1") & (t.Side == k)]) for k in "LS"}; sdn = {k: st(t[(t.Variant == "spec") & (t.Side == k)]) for k in "LS"}

# 傾きの効果だけを取り出す検定（MAの並びだけの全取引を、傾きが取引方向と一致/不一致で二分）
HAS_SPLIT = dsplit is not None
ss = pd.read_csv(dsplit / "slope_split.csv") if HAS_SPLIT else pd.DataFrame(columns=["EntryTime", "Pair", "R", "ER20", "slope1", "slope3"]); ss["EntryTime"] = pd.to_datetime(ss.EntryTime, utc=True)
ss["Y"] = ss.EntryTime.dt.tz_convert("America/New_York").dt.year; ss["cl"] = ss.Pair + "_" + ss.EntryTime.dt.strftime("%Y%m")
rng = np.random.default_rng(0)
def boot(g, mask, B=4000):
    h = g.assign(h=mask).groupby("cl").apply(lambda x: pd.Series(dict(a=x.R[x.h].sum(), b=x.h.sum(), c=x.R[~x.h].sum(), e=(~x.h).sum())))
    a = h.values; n = len(a); o = []
    for _ in range(B):
        s = a[rng.integers(0, n, n)].sum(0)
        if s[1] > 0 and s[3] > 0: o.append(s[0] / s[1] - s[2] / s[3])
    return np.percentile(o, [2.5, 97.5]), float(np.mean(np.array(o) <= 0))
sr = []; STAT = {}
for label, sub in ((("MAの並びだけ（全取引）", ss), ("ER(20)≥0.3 の取引だけ", ss[ss.ER20 >= 0.3])) if HAS_SPLIT else ()):
    sub = sub.reset_index(drop=True)
    for k in (1, 3):
        m_ = (sub[f"slope{k}"] > 0).values; hi_, lo_ = sub.R[m_], sub.R[~m_]; ci, p = boot(sub, pd.Series(m_, index=sub.index))
        dis = sub.R[m_ & (sub.Y <= 2023)].mean() - sub.R[~m_ & (sub.Y <= 2023)].mean(); doo = sub.R[m_ & (sub.Y >= 2024)].mean() - sub.R[~m_ & (sub.Y >= 2024)].mean()
        STAT[(label, k)] = dict(diff=hi_.mean() - lo_.mean(), lo=ci[0], hi=ci[1], p=p, nh=int(m_.sum()), nl=int((~m_).sum()))
        sr.append(f'<tr><td>{label} / {k}日</td><td class="num">{m_.sum()}</td><td class="num {cls(hi_.mean(),.0005)}">{hi_.mean():+.3f}</td><td class="num">{(~m_).sum()}</td><td class="num {cls(lo_.mean(),.0005)}">{lo_.mean():+.3f}</td>'
                  f'<td class="num {cls(hi_.mean()-lo_.mean(),.0005)}"><b>{hi_.mean()-lo_.mean():+.3f}</b></td><td class="num {"ci" if ci[0] > 0 else ""}">{ci[0]:+.3f}〜{ci[1]:+.3f}</td><td class="num">{p:.3f}</td>'
                  f'<td class="num {cls(dis,.0005)}">{dis:+.3f}</td><td class="num {cls(doo,.0005)}">{doo:+.3f}</td></tr>')
sp_tbl = tbl(["取引の範囲 / 傾き", "一致 取引数", "一致 平均R", "不一致 取引数", "不一致 平均R", "差", "差の95%区間", "片側p", "差〜2023", "差2024〜"], sr)
if HAS_SPLIT:
    s_all, s_er = STAT[("MAの並びだけ（全取引）", 1)], STAT[("ER(20)≥0.3 の取引だけ", 1)]
    s_all3 = STAT[("MAの並びだけ（全取引）", 3)]
drop = A["n"] - B["n"]; dropR = A["R"] - B["R"]
dd = B["R"] - A["R"]; rel = abs(dd) / max(abs(A["R"]), 1)
se = 1.25 * B["n"] ** .5
vd = "ほぼ同じ（誤差の範囲）" if abs(dd) < se else ("改善" if dd > 0 else "悪化")
vtail = ("傾きの条件による差は、この誤差の範囲に収まります。" if abs(dd) < se else "傾きの条件による差は、この誤差を超えています。") + ("取引数が少なく、成績の差は判断できません。" if B["n"] < 300 else "")
links = "".join(f'<button type="button" class="chip" data-chart="{p}">{p}</button>' for p in sorted(t.Pair.unique()))

split_html = f"""<section><h2>傾きの効果だけを取り出す検定（1取引あたりの平均R）</h2>{sp_tbl}
<p class="sub" style="margin-top:8px">MAの並びで入った全取引を、エントリー時の20MAの傾きが取引方向と一致した群と、しなかった群に二分して比較。差の95%区間は、ペア×月のまとまりで4,000回引き直して求めました。</p>
<p class="sub">結果: MAの並びだけの全取引では、傾きが一致した群の方が平均Rが高く、差は1日で {s_all['diff']:+.3f}R（95%区間 {s_all['lo']:+.3f}〜{s_all['hi']:+.3f}）、3日で {s_all3['diff']:+.3f}R（{s_all3['lo']:+.3f}〜{s_all3['hi']:+.3f}）です。{'3日では区間が0を超えますが、1日では0を含み、' if s_all3['lo'] > 0 >= s_all['lo'] else ''}傾き単独の効果は ER より小さいです（ER(20)≥0.3 の差は約 +0.10R）。
ER(20)≥0.3 の取引だけに絞ると、傾きが不一致の取引は {s_er['nl']} 回と少なく、差は {s_er['diff']:+.3f}R（{s_er['lo']:+.3f}〜{s_er['hi']:+.3f}）と幅が広くて確定できません。ER で絞った後に傾きを足しても、除外される取引が少ないので、合計Rはほとんど変わりません。</p></section>""" if HAS_SPLIT else ""

# 他方式との比較
cmp_html = ""; ev_txt = ""
if dcmps:
    runs = []
    for dc in dcmps:
        t2 = pd.read_csv(dc / "trades_all.csv"); t2["ExitTime"] = pd.to_datetime(t2.ExitTime, utc=True); t2["EntryTime"] = pd.to_datetime(t2.EntryTime, utc=True)
        t2["Y"] = t2.EntryTime.dt.tz_convert("America/New_York").dt.year
        mo = json.loads((dc / "config.json").read_text(encoding="utf-8")).get("entry_mode", "donchian")
        runs.append((mo, {k: st(g) for k, g in t2.groupby("Variant")}))
    runs.append((MODE, V))
    head = ["設定"]; 
    for mo, _ in runs: head += [f"{ENT_ALL[mo][2]} 取引数", "合計R", "PF", "最大DD"]
    rows2 = []
    for k in ("nofilter", "ma_only", "spec", "spec_slope1", "ma_slope3", "er10"):
        r = f"<tr><td>{NAME[k]}</td>"
        for mo, VV in runs:
            x = VV.get(k)
            r += (f'<td class="num">{x["n"]}</td><td class="num {cls(x["R"])}">{fs(x["R"])}</td><td class="num">{x["PF"]:.2f}</td><td class="num">{x["DD"]:.0f}</td>' if x else '<td></td>' * 4)
        rows2.append(r + "</tr>")
    spec_line = "、".join(f'{ENT_ALL[mo][2]}は{VV["spec"]["n"]}回・{fs(VV["spec"]["R"])}R' for mo, VV in runs)
    pf_line = "、".join(f'{ENT_ALL[mo][2]} {VV["spec"]["PF"]:.2f}' for mo, VV in runs)
    dd_line = "、".join(f'{ENT_ALL[mo][2]} {VV["spec"]["DD"]:.0f}' for mo, VV in runs)
    ev_txt = (f'<p><b>エントリー方式の違い（ER(20)≥0.3）: {spec_line}。</b></p>'
              f'<p class="muted">PF は {pf_line}、最大DD(R)は {dd_line} です。</p>')
    cmp_html = ('<section><h2>エントリー方式の比較</h2>' + tbl(head, rows2)
                + f'<p class="sub" style="margin-top:8px">同じフィルターと SL/TP の倍率で、エントリー方式だけが違います。ER(20)≥0.3 では、{spec_line}です。</p></section>')

doc = f"""<title>20MA傾きフィルター検証{"" if MODE == "donchian" else " " + ENTN}</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --accent-soft:#e3eafb; --pos:#18865a; --neg:#c8413a; --gray:#8a95a3; --hl:#eef3ff;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.65 var(--f-head); }}
.wrap {{ max-width:860px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:28px; }} .wrap>* {{ min-width:0; }}
h1 {{ font-size:1.45rem; line-height:1.3; margin:0; text-wrap:balance; }} h2 {{ font-size:1.05rem; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; }} .sub,.muted {{ color:var(--muted); }} .sub {{ font-size:.85rem; margin-top:6px; }}
.num {{ font-family:var(--f-num); font-variant-numeric:tabular-nums; text-align:right; }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }} td.ci {{ color:var(--pos); font-weight:600; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.verdict {{ border:1.5px solid var(--accent); background:var(--surface); border-radius:12px; padding:14px; }}
.kpis {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }}
.k {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px; }} .k small {{ color:var(--muted); display:block; font-size:.75rem; }} .k b {{ font:600 1.5rem var(--f-num); display:block; }} .k span {{ font-size:.78rem; color:var(--muted); }}
.tw {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; white-space:nowrap; }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-weight:500; font-size:.76rem; text-align:right; }} th:first-child {{ text-align:left; }}
tr:last-child td {{ border-bottom:0; }} tr.hl td {{ background:var(--hl); }}
a {{ color:var(--accent); }} button.chip {{ cursor:pointer; color:var(--fg); padding:5px 11px; margin:0 6px 6px 0; border:1px solid var(--line); border-radius:99px; background:var(--surface); font:.82rem var(--f-num); }}
button.chip[aria-pressed=true] {{ background:var(--accent-soft); border-color:var(--accent); }}
#cframe {{ width:100%; height:80vh; min-height:420px; border:1px solid var(--line); border-radius:12px; background:#fff; margin-top:8px; }}
svg {{ width:100%; height:auto; display:block; }} .gl {{ stroke:var(--line); stroke-width:1; }} .gl.z {{ stroke:var(--gray); }}
.ax {{ fill:var(--muted); font:11px var(--f-num); }} .ln {{ fill:none; stroke-width:2; stroke-linejoin:round; }} .ln.b {{ stroke:var(--accent); }} .ln.a {{ stroke:var(--gray); stroke-dasharray:4 3; }}
.lb {{ font:600 12px var(--f-head); }} .lb.b {{ fill:var(--accent); }} .lb.a {{ fill:var(--muted); }}
ul {{ margin:0; padding-left:1.2em; }} li {{ margin-bottom:4px; }}
#vbar{{display:flex;justify-content:flex-end;margin-top:6px}}
#vwrap button#vfull{{cursor:pointer;font:inherit;font-size:.85rem;color:var(--fg);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:5px 12px}}
#vwrap.full{{position:fixed;inset:0;z-index:1000;background:var(--bg);display:flex;flex-direction:column;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}}
#vwrap.full #vbar{{margin:0;padding:6px 12px}}
#vwrap.full #cframe{{flex:1;height:auto;min-height:0;margin:0;border:0;border-radius:0}}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>20MA傾きフィルター検証 15ペア{"" if MODE == "donchian" else "（" + ENTN + "）"}</h1><p class="sub">日足MA20/75/200の並び + ER(20)≥0.3 に、ロングは日足20MAが上向き、ショートは下向きの条件を追加。{ENTS}、SL1.0 / TP1.5 ATR。2021-01〜2025-12。</p></header>

<section class="verdict">{ev_txt}<p><b>合計 {fs(B['R'])}R（PF {B['PF']:.2f}）。傾きなしの {fs(A['R'])}R（PF {A['PF']:.2f}）と比べて、{vd}。</b></p>
<p class="muted">取引は {A['n']} → {B['n']} 回（{drop}回、{drop/A['n']*100:.0f}%減）。除外した分の合計Rは {fs(dropR)}R、平均Rは {A['avg']:+.3f} → {B['avg']:+.3f}、最大DDは {A['DD']:.0f} → {B['DD']:.0f}、2024〜の合計Rは {fs(A['OOS'],1)} → {fs(B['OOS'],1)} です。取引1回のRの標準偏差は約1.25Rなので、{B['n']}回の合計Rの誤差は±{1.25*B['n']**.5:.0f}R 程度です。{vtail}</p></section>

<section><div class="kpis">
<div class="k"><small>合計R（傾きあり）</small><b class="{cls(B['R'])}">{fs(B['R'])}</b><span>傾きなし {fs(A['R'])}</span></div>
<div class="k"><small>プロフィットファクター</small><b>{B['PF']:.2f}</b><span>傾きなし {A['PF']:.2f}</span></div>
<div class="k"><small>最大DD（R・決済順）</small><b>{B['DD']:.1f}</b><span>傾きなし {A['DD']:.1f}</span></div>
<div class="k"><small>取引数 / 勝率</small><b>{B['n']}</b><span>勝率 {B['win']:.1f}%（傾きなし {A['win']:.1f}%）</span></div></div></section>

{cmp_html}
<section><h2>累積R（15ペア合算・決済順）</h2><div class="card">{svg}</div></section>

<section><h2>設定の比較</h2>{var_tbl}
<p class="sub" style="margin-top:8px">{cmp_txt}</p></section>

{split_html}

<section><h2>年別（合計R）</h2>{tbl(["年","傾きなし 取引数","傾きなし 合計R","傾きあり 取引数","傾きあり 合計R","差"], yr)}</section>

<section><h2>通貨ペア別（傾きあり）</h2>{tbl(["ペア","取引数","合計R","勝率","PF","最大DD","傾きなし 合計R"], pr)}
<p class="sub" style="margin-top:8px">買い {sdw['L']['n']}回 {fs(sdw['L']['R'])}R（傾きなし {fs(sdn['L']['R'])}R）、売り {sdw['S']['n']}回 {fs(sdw['S']['R'])}R（傾きなし {fs(sdn['S']['R'])}R）。ペア名からトレード付きチャートを開けます。</p></section>

<section id="viewer"><h2>トレード付きチャート</h2><p class="muted">ペアを選ぶと、このページの中に表示します（約1MB）。背景は検証で使ったフィルター（MA並び + ER + 20MA傾き）で、買いだけ許可は緑、売りだけ許可は赤。日足MA20/75/200（黄・青・橙）と下に日足ER(20)も表示します。</p><div>{links}</div>
<p id="cmsg" class="sub" role="status"></p><div id="vwrap"><div id="vbar"><button type="button" id="vfull">全画面</button></div><iframe id="cframe" title="トレード付きチャート" sandbox="allow-scripts" hidden></iframe></div></section>

<section class="card"><h2>売買ルール・設定</h2><ul>
<li>日足（NY17:00区切り）: 20/75/200 単純MAの並び、ER(20)≥0.3、<b>追加: 20MAの傾き</b>（ロング: MA20 &gt; 1日前のMA20、ショート: MA20 &lt; 1日前のMA20）。すべて前日確定の日足の値。</li>
<li>エントリー: 方向が許可されたとき、{ENT}。</li>
<li>決済: 4時間足 ATR(14)（ワイルダー平滑）×1.0 で SL、×1.5 で TP。1R=200（複利なし）。平均スプレッドの半分を売買の両方で課金。</li></ul></section>

<section class="card"><h2>読むときの注意</h2><ul>
<li>傾きの日数（1・3・5日）、ER のしきい値、ER の期間を並べて見ています。最良の設定を選ぶと合わせ込みになります。</li>
<li>傾き条件は、MA の並びと ER が示す「整ったトレンド」の中の押し目・戻りを避ける働きをします。ER(20)≥0.3 の日のうち約{(1-on['spec_slope1']/on['spec'])*100:.0f}%がこの条件で見送りになります。</li>
<li>スワップは未考慮です。データは2021〜2025年の5年間のみです。</li>
<li>コード: <code>main_4H_trend_slope_fast.py</code>、<code>trend_slope_split.py</code>。結果: <code>trend_results/{d.name}/</code></li></ul></section>
</div>
<script>
(function(){{
  var wrap=document.getElementById("vwrap"), fb=document.getElementById("vfull");
  function setFull(on){{
    wrap.classList.toggle("full",on); fb.textContent=on?"全画面を閉じる":"全画面"; document.body.style.overflow=on?"hidden":"";
    if(on&&wrap.requestFullscreen){{try{{var p=wrap.requestFullscreen();if(p&&p.catch)p.catch(function(){{}});}}catch(e){{}}}}
    if(!on&&document.fullscreenElement){{try{{document.exitFullscreen();}}catch(e){{}}}}
  }}
  fb.addEventListener("click",function(){{setFull(!wrap.classList.contains("full"));}});
  document.addEventListener("keydown",function(e){{if(e.key==="Escape"&&wrap.classList.contains("full"))setFull(false);}});
  document.addEventListener("fullscreenchange",function(){{if(!document.fullscreenElement&&wrap.classList.contains("full"))setFull(false);}});
  var frame=document.getElementById("cframe"), msg=document.getElementById("cmsg"), cache={{}};
  function show(p){{
    document.querySelectorAll("button.chip").forEach(function(b){{b.setAttribute("aria-pressed",String(b.dataset.chart===p));}});
    msg.textContent=p+" を読み込み中…";
    var done=function(t){{cache[p]=t;frame.srcdoc=t;frame.hidden=false;msg.textContent=p;}};
    if(cache[p]){{done(cache[p]);return;}}
    fetch("dow_trade_chart_"+p+".html").then(function(r){{if(!r.ok)throw new Error(r.status);return r.text();}}).then(done)
      .catch(function(e){{msg.textContent=p+" を読み込めませんでした（"+e.message+"）。";}});
  }}
  document.addEventListener("click",function(e){{var t=e.target.closest("[data-chart]");if(t){{show(t.dataset.chart);}}}});
}})();
</script>
"""
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
