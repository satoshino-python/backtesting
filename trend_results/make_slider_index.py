"""make_er_slider_charts.py の結果フォルダに、チャートを開くための report.html を作る。使い方: python trend_results/make_slider_index.py <実行フォルダ>"""
import sys
from pathlib import Path
import pandas as pd
d = Path(sys.argv[1]); s = pd.read_csv(d / "summary.csv")
btn = " ".join(f'<button type="button" class="chip" data-chart="{r.Pair}">{r.Pair}<small>{r.Trades}回</small></button>' for r in s.itertuples())
doc = """<title>ERスライダー付きチャート</title>
<style>
:root { --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --accent-soft:#e3eafb; --pos:#18865a; --neg:#c8413a;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,Menlo,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; color-scheme:dark; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:15px/1.65 var(--f-head); }
.wrap { max-width:960px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:24px; } .wrap>* { min-width:0; }
h1 { font-size:1.45rem; line-height:1.3; margin:0; text-wrap:balance; } h2 { font-size:1.05rem; margin:0 0 10px; }
p { margin:0 0 8px; } .sub,.muted { color:var(--muted); } .sub { font-size:.85rem; margin-top:6px; }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }
button.chip { cursor:pointer; color:var(--fg); display:inline-flex; flex-direction:column; align-items:center; gap:0; padding:6px 12px; margin:0 6px 6px 0; border:1px solid var(--line); border-radius:12px; background:var(--surface); font:.85rem var(--f-num); }
button.chip small { color:var(--muted); font-size:.68rem; }
button.chip[aria-pressed=true] { background:var(--accent-soft); border-color:var(--accent); }
#cframe { width:100%; height:86vh; min-height:480px; border:1px solid var(--line); border-radius:12px; background:#111418; margin-top:8px; }
ul { margin:0; padding-left:1.2em; } li { margin-bottom:4px; }
#vbar{display:flex;justify-content:flex-end;margin-top:6px}
#vwrap button#vfull{cursor:pointer;font:inherit;font-size:.85rem;color:var(--fg);background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:5px 12px}
#vwrap.full{position:fixed;inset:0;z-index:1000;background:var(--bg);display:flex;flex-direction:column;padding:env(safe-area-inset-top,0px) 0 env(safe-area-inset-bottom,0px)}
#vwrap.full #vbar{margin:0;padding:6px 12px}
#vwrap.full #cframe{flex:1;height:auto;min-height:0;margin:0;border:0;border-radius:0}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<div class="wrap">
<header><h1>ERスライダー付きチャート</h1><p class="sub">日足MAの並び（20&gt;75&gt;200 / 逆）だけで入った全取引を表示し、ER の期間としきい値をスライダーで動かして、フィルターを通る取引と除外される取引を目で確かめる。15ペア、2021〜2025、4H 18本ブレイク SL1.0 / TP1.5 ATR。</p></header>

<section class="card"><h2>見方</h2><ul>
<li><b>ER期間・ERしきい値</b>のスライダーを動かすと、背景（緑=買いだけ許可、赤=売りだけ許可、無色=見送り）、日足ERのパネル、取引の濃さが変わります。</li>
<li>フィルターを<b>通る取引は濃く</b>、<b>除外される取引は薄く</b>表示します（「除外された取引も薄く表示」を外すと消えます）。</li>
<li>設定バーの緑・赤の数字は、通過した取引と除外された取引の件数・平均R・合計Rで、スライダーに合わせて更新されます。</li>
<li>トレードにカーソルを合わせると、エントリー時の日足ERとフィルターの通過/除外を表示します。◀▶ キーで前後のトレードに移動します。</li>
<li>日足MA（黄=20、青=75、橙=200）は前日に確定した値で、検証と同じです。エントリーラインは直近N本の最高値/最安値です。</li>
<li>感覚と合っているかを見るには、「ここは強いトレンドだ」と思う区間で、背景の緑/赤が付いて、勝つ取引が通過しているかを確かめてください。</li></ul></section>

<section><h2>ペアを選ぶ（括弧は MAの並びだけの取引数）</h2><div>__BTN__</div>
<p id="cmsg" class="sub" role="status"></p><div id="vwrap"><div id="vbar"><button type="button" id="vfull">全画面</button></div><iframe id="cframe" title="ERスライダー付きチャート" sandbox="allow-scripts" hidden></iframe></div></section>

<section class="card"><h2>注意</h2><ul>
<li>ペアごとの取引は200〜300回と少なく、通過/除外の平均Rの差は、1ペアだけではぶれが大きいです（全ペアの検定結果は別のレポートを参照）。</li>
<li>表示する取引は、MAの並びだけで入ったものです。フィルターを入れて検証した取引とは、除外した分の代わりに別の取引が入る点で違います。</li>
<li>ER期間 2〜60、しきい値 0〜1 の範囲で動かせます。しきい値 0 は MA の並びだけです。</li></ul></section>
</div>
<script>
(function(){
  var wrap=document.getElementById("vwrap"), fb=document.getElementById("vfull");
  function setFull(on){
    wrap.classList.toggle("full",on); fb.textContent=on?"全画面を閉じる":"全画面"; document.body.style.overflow=on?"hidden":"";
    if(on&&wrap.requestFullscreen){try{var p=wrap.requestFullscreen();if(p&&p.catch)p.catch(function(){});}catch(e){}}
    if(!on&&document.fullscreenElement){try{document.exitFullscreen();}catch(e){}}
  }
  fb.addEventListener("click",function(){setFull(!wrap.classList.contains("full"));});
  document.addEventListener("keydown",function(e){if(e.key==="Escape"&&wrap.classList.contains("full"))setFull(false);});
  document.addEventListener("fullscreenchange",function(){if(!document.fullscreenElement&&wrap.classList.contains("full"))setFull(false);});
  var frame=document.getElementById("cframe"), msg=document.getElementById("cmsg"), cache={};
  function show(p){
    document.querySelectorAll("button.chip").forEach(function(b){b.setAttribute("aria-pressed",String(b.dataset.chart===p));});
    msg.textContent=p+" を読み込み中…";
    var done=function(t){cache[p]=t;frame.srcdoc=t;frame.hidden=false;msg.textContent=p+"（スライダーは設定バーの中）";};
    if(cache[p]){done(cache[p]);return;}
    fetch("er_slider_chart_"+p+".html").then(function(r){if(!r.ok)throw new Error(r.status);return r.text();}).then(done)
      .catch(function(e){msg.textContent=p+" を読み込めませんでした（"+e.message+"）。";});
  }
  document.addEventListener("click",function(e){var t=e.target.closest("[data-chart]");if(t){show(t.dataset.chart);}});
})();
</script>
""".replace("__BTN__", btn)
(d / "report.html").write_text(doc, encoding="utf-8"); print("ok")
