"""trend_results/<実行>/ の trades_all.csv から report.html を作る。使い方: python trend_results/make_report.py <実行フォルダ>"""
import sys, html
from pathlib import Path
import numpy as np, pandas as pd
d = Path(sys.argv[1])
t = pd.read_csv(d / "trades_all.csv")
t["ExitTime"] = pd.to_datetime(t.ExitTime, utc=True); t["EntryTime"] = pd.to_datetime(t.EntryTime, utc=True)
t["Year"] = t.EntryTime.dt.tz_convert("America/New_York").dt.year
on = pd.read_csv(d / "filter_on_share.csv").query("Variant=='spec'").FilterOnPct.mean()

def st(g):
    g = g.sort_values("ExitTime"); r = g.R.values; eq = np.cumsum(r)
    dd = float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(r) else 0.0
    gl = -r[r < 0].sum()
    return dict(n=len(r), R=r.sum(), avg=r.mean() if len(r) else 0, win=(r > 0).mean() * 100 if len(r) else 0,
                PF=r[r > 0].sum() / gl if gl > 0 else float("nan"), DD=dd)
def cls(x): return "pos" if x > 0 else "neg" if x < 0 else ""
def f(x, p=1, sign=False): return f"{x:+.{p}f}" if sign else f"{x:.{p}f}"
def tbl(head, rows):
    h = "".join(f"<th>{c}</th>" for c in head)
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'

spec, nof = t[t.Variant == "spec"], t[t.Variant == "nofilter"]
S, N = st(spec), st(nof)
IS = st(spec[spec.Year <= 2023]); OOS = st(spec[spec.Year >= 2024])
NIS = st(nof[nof.Year <= 2023]); NOOS = st(nof[nof.Year >= 2024])

# 損益曲線（決済順の累積R、週ごと）
def curve(g):
    g = g.sort_values("ExitTime"); s = g.set_index("ExitTime").R.cumsum()
    return s.resample("W").last().ffill()
cs, cn = curve(spec), curve(nof)
idx = cs.index.union(cn.index); cs, cn = cs.reindex(idx).ffill().fillna(0), cn.reindex(idx).ffill().fillna(0)
W, H, L, B, T = 760, 300, 46, 28, 12
lo, hi = min(cs.min(), cn.min(), 0), max(cs.max(), cn.max(), 0)
step = 50; lo = np.floor(lo / step) * step; hi = np.ceil(hi / step) * step
x = lambda i: L + (W - L - 10) * i / (len(idx) - 1)
y = lambda v: T + (H - T - B) * (hi - v) / (hi - lo)
grid = "".join(f'<line x1="{L}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="gl{" z" if v==0 else ""}"/><text x="{L-6}" y="{y(v)+4:.1f}" class="ax" text-anchor="end">{int(v)}</text>'
                for v in np.arange(lo, hi + 1, step))
yrs = "".join(f'<text x="{x(i):.1f}" y="{H-8}" class="ax" text-anchor="middle">{ts.year}</text>'
              for i, ts in enumerate(idx) if i and ts.year != idx[i-1].year)
path = lambda s: "M" + " L".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s.values))
svg = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="累積Rの推移"><g>{grid}{yrs}</g>'
       f'<path d="{path(cn)}" class="ln n"/><path d="{path(cs)}" class="ln s"/>'
       f'<text x="{x(len(idx)-1)-4:.1f}" y="{y(cs.iloc[-1])-8:.1f}" class="lb s" text-anchor="end">添付ルール {cs.iloc[-1]:+.0f}R</text>'
       f'<text x="{x(len(idx)-1)-4:.1f}" y="{y(cn.iloc[-1])+16:.1f}" class="lb n" text-anchor="end">フィルターなし {cn.iloc[-1]:+.0f}R</text></svg>')

# 年別
yr = []
for y_ in range(2021, 2026):
    a, b = st(spec[spec.Year == y_]), st(nof[nof.Year == y_])
    yr.append(f'<tr><td>{y_}</td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{f(a["R"],1,1)}</td><td class="num">{a["win"]:.0f}%</td>'
              f'<td class="num">{b["n"]}</td><td class="num {cls(b["R"])}">{f(b["R"],1,1)}</td></tr>')
# ペア別
prow = []
pst = {p: st(g) for p, g in spec.groupby("Pair")}
for p in sorted(pst, key=lambda k: -pst[k]["R"]):
    a = pst[p]; b = st(nof[nof.Pair == p])
    prow.append(f'<tr><td><a href="#viewer" data-chart="{p}">{p}</a></td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{f(a["R"],1,1)}</td>'
                f'<td class="num">{a["win"]:.0f}%</td><td class="num">{a["PF"]:.2f}</td><td class="num">{a["DD"]:.1f}</td><td class="num {cls(b["R"])}">{f(b["R"],0,1)}</td></tr>')
npos = sum(1 for a in pst.values() if a["R"] > 0)
jpy = sum(a["R"] for p, a in pst.items() if p.endswith("JPY"))
# 売買別
sd = {s: st(spec[spec.Side == s]) for s in "LS"}
# 感度
names = {"spec": "添付ルール（ER0.3 / 18本 / SL1.0 TP1.5）", "nofilter": "フィルターなし", "er0.25": "ER閾値 0.25", "er0.35": "ER閾値 0.35",
         "er0.4": "ER閾値 0.4", "brk12": "ブレイク 12本", "brk24": "ブレイク 24本", "brk30": "ブレイク 30本",
         "sl0.8_tp1.5": "SL 0.8 / TP 1.5", "sl1.5_tp1.5": "SL 1.5 / TP 1.5", "sl1.0_tp1.0": "SL 1.0 / TP 1.0",
         "sl1.0_tp2.0": "SL 1.0 / TP 2.0", "sl1.0_tp3.0": "SL 1.0 / TP 3.0", "sl1.5_tp3.0": "SL 1.5 / TP 3.0"}
srow = []
for k, nm in names.items():
    g = t[t.Variant == k]; a = st(g); i = st(g[g.Year <= 2023]); o = st(g[g.Year >= 2024])
    srow.append(f'<tr{" class=hl" if k=="spec" else ""}><td>{nm}</td><td class="num">{a["n"]}</td><td class="num {cls(a["R"])}">{f(a["R"],1,1)}</td>'
                f'<td class="num">{a["PF"]:.2f}</td><td class="num">{a["DD"]:.0f}</td><td class="num {cls(i["R"])}">{f(i["R"],0,1)}</td><td class="num {cls(o["R"])}">{f(o["R"],0,1)}</td></tr>')
links = " ".join(f'<button type="button" class="chip" data-chart="{p}">{p}</button>' for p in sorted(pst))

SCRIPT = r"""
<script>
(function(){
  var frame=document.getElementById("cframe"), msg=document.getElementById("cmsg"), cache={};
  function show(p){
    document.querySelectorAll("button.chip").forEach(function(b){b.setAttribute("aria-pressed",String(b.dataset.chart===p));});
    msg.textContent=p+" を読み込み中…";
    var done=function(t){cache[p]=t;frame.srcdoc=t;frame.hidden=false;msg.textContent=p+"（検証で使ったフィルターを背景に表示）";};
    if(cache[p]){done(cache[p]);return;}
    fetch("dow_trade_chart_"+p+".html").then(function(r){if(!r.ok)throw new Error(r.status);return r.text();}).then(done)
      .catch(function(e){msg.textContent=p+" を読み込めませんでした（"+e.message+"）。";});
  }
  document.addEventListener("click",function(e){var t=e.target.closest("[data-chart]");if(t){show(t.dataset.chart);}});
})();
</script>
"""

doc = f"""<title>トレンドブレイクアウト検証</title>
<style>
:root {{ --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --accent-soft:#e3eafb;
  --pos:#18865a; --neg:#c8413a; --gray:#8a95a3; --hl:#eef3ff;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae;
  --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae;
  --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gray:#7d8997; --hl:#1b2638; color-scheme:dark; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.65 var(--f-head); }}
.wrap {{ max-width:860px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:28px; }} .wrap>* {{ min-width:0; }}
h1 {{ font-size:1.45rem; line-height:1.3; margin:0; text-wrap:balance; }} h2 {{ font-size:1.05rem; margin:0 0 10px; text-wrap:balance; }}
p {{ margin:0 0 8px; }} .sub,.muted {{ color:var(--muted); }} .sub {{ font-size:.85rem; margin-top:6px; }}
.num {{ font-family:var(--f-num); font-variant-numeric:tabular-nums; text-align:right; }} .pos {{ color:var(--pos); }} .neg {{ color:var(--neg); }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }}
.kpis {{ display:grid; grid-template-columns:repeat(2,1fr); gap:10px; }}
.k {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:12px; }}
.k small {{ color:var(--muted); display:block; font-size:.75rem; }} .k b {{ font:600 1.5rem var(--f-num); display:block; }} .k span {{ font-size:.78rem; color:var(--muted); }}
.tw {{ overflow-x:auto; background:var(--surface); border:1px solid var(--line); border-radius:12px; }}
table {{ border-collapse:collapse; width:100%; font-size:.85rem; white-space:nowrap; }}
th,td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; }} th {{ color:var(--muted); font-weight:500; font-size:.76rem; }}
td.num,th:not(:first-child) {{ text-align:right; }} tr:last-child td {{ border-bottom:0; }} tr.hl td {{ background:var(--hl); }}
a {{ color:var(--accent); }} button.chip {{ cursor:pointer; color:var(--fg); }} button.chip[aria-pressed=true] {{ background:var(--accent-soft); border-color:var(--accent); }} #cframe {{ width:100%; height:80vh; min-height:420px; border:1px solid var(--line); border-radius:12px; background:#fff; margin-top:8px; }} .chip {{ display:inline-block; padding:5px 11px; margin:0 6px 6px 0; border:1px solid var(--line); border-radius:99px; background:var(--surface); text-decoration:none; font:.82rem var(--f-num); }}
svg {{ width:100%; height:auto; display:block; }} .gl {{ stroke:var(--line); stroke-width:1; }} .gl.z {{ stroke:var(--gray); }}
.ax {{ fill:var(--muted); font:11px var(--f-num); }} .ln {{ fill:none; stroke-width:2; stroke-linejoin:round; }}
.ln.s {{ stroke:var(--accent); }} .ln.n {{ stroke:var(--gray); stroke-dasharray:4 3; }} .lb {{ font:600 12px var(--f-head); }} .lb.s {{ fill:var(--accent); }} .lb.n {{ fill:var(--muted); }}
ul {{ margin:0; padding-left:1.2em; }} li {{ margin-bottom:4px; }}
.verdict {{ border:1.5px solid var(--accent); background:var(--surface); border-radius:12px; padding:14px; }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>トレンドブレイクアウト検証 15ペア</h1>
<p class="sub">日足MA20/75/200 + ER(20)≥0.3 で方向を絞り、4時間足18本ブレイクで入る。SL ATR×1.0 / TP ATR×1.5。2021-01〜2025-12、GMOクリック1分足執行。</p></header>

<section class="verdict"><p><b>結果: 合計 {S['R']:+.0f}R（PF {S['PF']:.2f}）で、フィルターなしの {N['R']:+.0f}R より大きく改善。ただし直近2年は横ばい。</b></p>
<p class="muted">フィルターが効いた期間は全体の約{on:.0f}%と短く、1ペア年約19回しか入りません。損益の大半は円クロスと買い側で、2024〜2025年（検証の後半）は {OOS['R']:+.1f}R でした。</p></section>

<section><div class="kpis">
<div class="k"><small>合計R（添付ルール）</small><b class="{cls(S['R'])}">{S['R']:+.1f}</b><span>フィルターなし {N['R']:+.1f}</span></div>
<div class="k"><small>プロフィットファクター</small><b>{S['PF']:.2f}</b><span>フィルターなし {N['PF']:.2f}</span></div>
<div class="k"><small>最大DD（R・全ペア決済順）</small><b>{S['DD']:.1f}</b><span>フィルターなし {N['DD']:.0f}</span></div>
<div class="k"><small>取引数 / 勝率</small><b>{S['n']}</b><span>勝率 {S['win']:.1f}%（分岐点約40%）/ 1取引 {S['avg']:+.3f}R</span></div>
</div></section>

<section><h2>累積Rの推移（15ペア合算・決済順）</h2><div class="card">{svg}</div>
<p class="sub">合計Rは1Rを初期資金の2%（200ドル固定）としたR倍数。合計Rを1ペア分に直すと {S['R']:.0f}R×2%÷15ペア ≒ 1ペアあたり約+{S['R']*2/15:.0f}%（5年）の目安。</p></section>

<section><h2>前半・後半（エントリー年で分割）</h2>
{tbl(["","取引数","合計R","PF","勝率"],[
f'<tr><td>添付ルール 2021〜2023</td><td class="num">{IS["n"]}</td><td class="num {cls(IS["R"])}">{IS["R"]:+.1f}</td><td class="num">{IS["PF"]:.2f}</td><td class="num">{IS["win"]:.1f}%</td></tr>',
f'<tr><td>添付ルール 2024〜2025</td><td class="num">{OOS["n"]}</td><td class="num {cls(OOS["R"])}">{OOS["R"]:+.1f}</td><td class="num">{OOS["PF"]:.2f}</td><td class="num">{OOS["win"]:.1f}%</td></tr>',
f'<tr><td>フィルターなし 2021〜2023</td><td class="num">{NIS["n"]}</td><td class="num {cls(NIS["R"])}">{NIS["R"]:+.1f}</td><td class="num">{NIS["PF"]:.2f}</td><td class="num">{NIS["win"]:.1f}%</td></tr>',
f'<tr><td>フィルターなし 2024〜2025</td><td class="num">{NOOS["n"]}</td><td class="num {cls(NOOS["R"])}">{NOOS["R"]:+.1f}</td><td class="num">{NOOS["PF"]:.2f}</td><td class="num">{NOOS["win"]:.1f}%</td></tr>'])}
<p class="sub" style="margin-top:8px">パラメータはルール通りで固定し、途中で選び直していません。前半の成績は、ルールを前半で選んだものではありません。後半の弱さが相場の変化か偶然かは、この結果だけでは分かりません。</p></section>

<section><h2>年別（合計R）</h2>
{tbl(["年","取引数","合計R","勝率","なし取引数","なし合計R"], yr)}</section>

<section><h2>通貨ペア別（添付ルール）</h2>
{tbl(["ペア","取引数","合計R","勝率","PF","最大DD","なし合計R"], prow)}
<p class="sub" style="margin-top:8px">プラスは{npos}/15ペア。円クロス・円ペアの合計は {jpy:+.0f}R で、全体の大半を占めます。買い {sd['L']['n']}回 {sd['L']['R']:+.0f}R、売り {sd['S']['n']}回 {sd['S']['R']:+.0f}R。ペア名からトレード付きチャートを開けます。</p></section>

<section id="viewer"><h2>トレード付きチャート</h2><p class="muted">ペアを選ぶと、このページの中に表示します（約3MBを読み込みます）。エントリー▲▼・決済●・SL/TP、背景は「検証で使ったフィルター」（買いだけ/売りだけ/見送り）。</p><div>{links}</div>
<p id="cmsg" class="sub" role="status"></p><iframe id="cframe" title="トレード付きダウ理論チャート" sandbox="allow-scripts" hidden></iframe></section>

<section><h2>パラメータ感度（一度に1つだけ変更）</h2>
{tbl(["設定","取引数","合計R","PF","最大DD","〜2023","2024〜"], srow)}
<p class="sub" style="margin-top:8px">ER閾値は0.25〜0.4のどれもプラス（+54〜+103R）で、閾値を上げるほど取引数が減ります。1取引あたりは0.04〜0.07Rで大きな差はありません。ブレイク本数は24本が最良、12本は弱い。TP/SL比は 1:1 だとほぼ ±0。TP3.0 までプラスなので、1.5固定が最適というわけではありません。全期間を見て並べた表なので、最良値を選ぶのは合わせ込みになります。</p></section>

<section class="card"><h2>売買ルール・設定</h2><ul>
<li>日足（NY17:00区切り）: 20/75/200 単純MAの並びと ER(20)=|C−C[−20]|/Σ|ΔC| ≥ 0.3。前日確定の日足だけを使用。</li>
<li>エントリー: 方向が許可されたとき、確定済み直近18本の4時間足の最高値/最安値への逆指値。判定中の足は含めない。</li>
<li>決済: 4時間足 ATR(14)（ワイルダー平滑）×1.0 で SL、×1.5 で TP。変更なし。保有中にフィルターが外れても保持。同じ1分足で両方に届けば SL 優先。</li>
<li>枚数: 1R=200（初期資金10,000の2%、複利なし）÷ SL幅。同一ペア同時保有1。ペアは別口座として扱う。</li>
<li>コスト: 平均スプレッドの半分を売買の両方で課金。週末の持ち越しあり。</li></ul></section>

<section class="card"><h2>読むときの注意</h2><ul>
<li>スワップは未考慮です。日足の上昇/下降トレンドに乗る保有は数日に及ぶので、キャリーの有利/不利が結果に出ます。円クロスの買いが強い点は、円安トレンドの影響を受けている可能性があります。</li>
<li>損益は決済通貨のまま200固定でRに換算しており、口座通貨建てへの換算は行っていません。R倍数としては影響しません。</li>
<li>フィルターONは全体の約{on:.0f}%の期間で、取引は年に1ペア約19回と少ない。ペアごとの差（−20R〜+42R）は偶然の幅が大きいと見るべきです。</li>
<li>チャートのエントリーラインはスイングの高値/安値で、今回の「直近18本の最高値/最安値」とは異なります。エントリー位置はトレードの▲▼で確認してください。ATRはワイルダー平滑で、既存の検証（EMA span）とは計算が異なります。</li>
<li>フィルターなしの比較は、同じ逆指値の仕組みで両方向を置きます。反対側の逆指値が残って既存ポジションの決済になる挙動（STRATEGIES.md 3章の注意）を含みます。</li>
<li>200MAの助走に2020年のデータを使い、取引は2021年以降です。2020年相場は含みません。</li>
<li>指標・コードは <code>main_4H_trend_breakout_fast.py</code>、結果は <code>trend_results/{d.name}/</code>。fast_engine は backtesting.py と一致確認済みですが、今回の入力（ブレイクアウト線）での再照合は行っていません。</li></ul></section>
</div>"""
(d / "report.html").write_text(doc + SCRIPT, encoding="utf-8")
print("ok", len(doc))
