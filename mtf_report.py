"""
mtf_pullback_strategy.py の結果ページ（report.html）を作る。見た目は fast_results の既存レポートに揃えている。
"""
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

CSS = """/* Layout: one phone-width column — headline numbers, equity curve, yearly / funnel / trade tables, chart links, rules and cautions. Same look as the existing swing-breakout reports */
:root { --bg:#f3f5f7; --surface:#fff; --line:#dde2e8; --fg:#17202b; --muted:#5e6b7a; --accent:#2a5bd7; --accent-soft:#e3eafb; --pos:#18865a; --neg:#c8413a; --gold:#9a7412;
  --f-head:"Zen Kaku Gothic New","Hiragino Sans","Noto Sans JP",system-ui,sans-serif; --f-num:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gold:#e0b84a; color-scheme:dark; } }
:root[data-theme="dark"] { --bg:#0f141a; --surface:#171e26; --line:#2a333e; --fg:#e3e8ee; --muted:#93a0ae; --accent:#79a2ff; --accent-soft:#1d2a45; --pos:#3ccb8b; --neg:#f0726a; --gold:#e0b84a; color-scheme:dark; }
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--fg); font:15px/1.6 var(--f-head); -webkit-text-size-adjust:100%; }
.wrap { max-width:860px; margin:0 auto; padding:20px 16px 48px; display:grid; gap:26px; }
.wrap > * { min-width:0; }
h1 { font-size:1.45rem; line-height:1.3; margin:0; font-weight:700; text-wrap:balance; }
h2 { font-size:1.05rem; margin:0 0 10px; font-weight:700; text-wrap:balance; }
p { margin:0; } .sub { color:var(--muted); font-size:.85rem; margin-top:6px; }
.num { font-family:var(--f-num); font-variant-numeric:tabular-nums; } .pos { color:var(--pos); } .neg { color:var(--neg); } .muted { color:var(--muted); }
.cmp { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; background:var(--line); border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.cmp div { background:var(--surface); padding:10px 12px; min-width:0; }
.cmp .l { font-size:.7rem; color:var(--muted); letter-spacing:.04em; }
.cmp .v { font-family:var(--f-num); font-size:1.2rem; font-weight:600; white-space:nowrap; }
.cmp .was { font-family:var(--f-num); font-size:.72rem; color:var(--muted); }
.card { background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; }
.card p { font-size:.88rem; color:var(--muted); } .card p + p { margin-top:6px; } .card b { color:var(--fg); font-weight:500; }
svg { width:100%; height:auto; display:block; }
.gl { stroke:var(--line); stroke-width:1; } .ax { fill:var(--muted); font:11px var(--f-num); }
.l-n { fill:none; stroke:var(--accent); stroke-width:2; } .l-g { fill:none; stroke:var(--gold); stroke-width:1.5; stroke-dasharray:4 3; }
.legend { display:flex; gap:16px; flex-wrap:wrap; font-size:.78rem; color:var(--muted); margin-top:6px; }
.legend i { display:inline-block; width:18px; height:0; border-top:2px solid var(--accent); vertical-align:middle; margin-right:6px; }
.legend i.g { border-top:2px dashed var(--gold); }
.scroll { overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--line); border-radius:12px; background:var(--surface); }
table { border-collapse:collapse; width:100%; font-size:.82rem; font-family:var(--f-num); }
th,td { padding:7px 10px; text-align:right; white-space:nowrap; border-bottom:1px solid var(--line); }
th { font-weight:500; color:var(--muted); font-size:.72rem; font-family:var(--f-head); }
th:first-child,td:first-child { text-align:left; } tr:last-child td { border-bottom:0; } tr.tot td { background:var(--accent-soft); font-weight:600; }
.link { display:block; text-decoration:none; color:var(--fg); background:var(--surface); border:1.5px solid var(--accent); border-radius:12px; padding:14px; }
.link:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }
.link b { display:block; } .link span { font-size:.82rem; color:var(--muted); }
ul { margin:0; padding-left:1.2em; display:grid; gap:6px; font-size:.88rem; color:var(--muted); } li b { color:var(--fg); font-weight:500; }
code { font-family:var(--f-num); font-size:.78rem; background:var(--accent-soft); padding:1px 5px; border-radius:4px; }
.hint { font-size:.8rem; margin-top:8px; } .card + .card { margin-top:10px; }
"""


def _f(v, d=2, sign=False):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def _cls(v):
    return "pos" if v > 0 else "neg" if v < 0 else ""


def equity_svg(times, r, width=640, height=240):
    curve = np.concatenate([[0.0], np.cumsum(r)])
    t = pd.to_datetime(pd.Series(times)).dt.tz_localize(None) if getattr(pd.Series(times).dt, "tz", None) is not None \
        else pd.to_datetime(pd.Series(times))
    if len(t) == 0:
        return ""
    t0, t1 = t.iloc[0] - pd.Timedelta(days=30), t.iloc[-1] + pd.Timedelta(days=30)
    L, R, T, B = 44, 12, 12, 26
    W, H = width - L - R, height - T - B
    lo, hi = min(curve.min(), 0), max(curve.max(), 0)
    step = 5 if hi - lo <= 40 else 10
    lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
    if hi == lo:
        hi = lo + step
    X = lambda ts: L + (ts - t0) / (t1 - t0) * W
    Y = lambda v: T + (hi - v) / (hi - lo) * H
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="累積R">']
    for v in np.arange(lo, hi + step / 2, step):
        out.append(f'<line x1="{L}" x2="{L + W}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="gl"/>'
                   f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" class="ax" text-anchor="end">{v:g}</text>')
    for y in range(t0.year + 1, t1.year + 1):
        x = X(pd.Timestamp(f"{y}-01-01"))
        out.append(f'<text x="{x:.1f}" y="{height - 6}" class="ax" text-anchor="middle">{y}</text>')
    xs = [X(t0 + pd.Timedelta(days=30))] + [X(v) for v in t]
    d = f"M{xs[0]:.1f},{Y(0):.1f}"
    for k in range(1, len(xs)):
        d += f" L{xs[k]:.1f},{Y(curve[k - 1]):.1f} L{xs[k]:.1f},{Y(curve[k]):.1f}"
    out.append(f'<path d="{d}" class="l-n"/>')
    out.append(f'<circle cx="{xs[-1]:.1f}" cy="{Y(curve[-1]):.1f}" r="3.5" fill="var(--accent)"/>')
    out.append("</svg>")
    return "".join(out)


def table(head, rows, total=None):
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td{(' class=' + chr(34) + c + chr(34)) if c else ''}>{v}</td>"
                                    for v, c in row) + "</tr>" for row in rows)
    if total:
        body += '<tr class="tot">' + "".join(f"<td>{v}</td>" for v, _ in total) + "</tr>"
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def write_report(path, trades, stats, stats_side, funnels, cfg, spread, chart_file, period_label):
    d = cfg.price_decimals
    r = trades["R"].to_numpy(float) if len(trades) else np.array([])
    pip = 10 ** -(d - 1)
    risk_pips = ((trades["entry"] - trades["sl"]).abs() / pip) if len(trades) else pd.Series(dtype=float)
    cost_r = (trades["R_gross"] - trades["R"]).mean() if len(trades) else float("nan")
    wins = r[r > 0]
    hold_h = ((pd.to_datetime(trades["exit_time"], utc=True) - pd.to_datetime(trades["entry_time"], utc=True))
              .dt.total_seconds() / 3600) if len(trades) else pd.Series(dtype=float)

    kpi = f"""<div class="cmp">
<div><div class="l">合計R</div><div class="v {_cls(stats['合計R'])}">{_f(stats['合計R'], 2, True)}</div><div class="was">平均 {_f(stats['平均R'], 3, True)}R / 取引</div></div>
<div><div class="l">プロフィットファクター</div><div class="v">{_f(stats['PF'])}</div><div class="was">勝ちR合計 ÷ 負けR合計</div></div>
<div><div class="l">最大DD（R）</div><div class="v">{_f(stats['最大DD [R]'], 1)}</div><div class="was">累積Rのピークからの下落</div></div>
<div><div class="l">取引数 / 勝率</div><div class="v">{stats['トレード数']} / {_f(stats['勝率 [%]'], 1)}%</div><div class="was">損益分岐の勝率 約 {100 / (1 + cfg.rr_target):.0f}%（TP {cfg.rr_target}R）</div></div>
<div><div class="l">損切り幅（中央値）</div><div class="v">{_f(risk_pips.median(), 1)} pips</div><div class="was">範囲 {_f(risk_pips.min(), 1)} ~ {_f(risk_pips.max(), 1)}</div></div>
<div><div class="l">保有時間（中央値）</div><div class="v">{_f(hold_h.median(), 1)} 時間</div><div class="was">スプレッド負担 平均 {_f(cost_r, 3)}R / 取引</div></div>
</div>""" if cfg.rr_target else ""

    if len(trades):
        yr = pd.to_datetime(trades["exit_time"], utc=True).dt.tz_convert("America/New_York").dt.year
        rows = []
        for y, g in trades.groupby(yr):
            s = g["R"]
            rows.append([(str(y), ""), (str(len(s)), ""), (f"{(s > 0).mean() * 100:.0f}%", ""),
                         (_f(s.sum(), 2, True), _cls(s.sum()))])
        yearly = table(["決済年", "取引", "勝率", "合計R"], rows,
                       [("合計", ""), (str(len(r)), ""), (f"{stats['勝率 [%]']:.0f}%", ""), (_f(r.sum(), 2, True), "")])
        trows = []
        SIDE = {"long": "買い", "short": "売り"}
        for k, t in trades.reset_index(drop=True).iterrows():
            et = pd.Timestamp(t["entry_time"])
            trows.append([(f'<a href="setups.html#t{k + 1}">{k + 1}</a>', ""), (SIDE[t["side"]], ""), (f"{et:%Y-%m-%d %H:%M}", ""),
                          (f"{t['entry']:.{d}f}", ""), (f"{t['sl']:.{d}f}", ""), (f"{t['tp']:.{d}f}", ""),
                          (f"{t['rr_level']:.{d}f}", ""), (t["exit_reason"], ""), (_f(t["R"], 2, True), _cls(t["R"]))])
        trade_tbl = table(["#", "売買", "エントリー（NY）", "価格", "SL", "TP", "H_b", "決済", "R"], trows)
    else:
        yearly = trade_tbl = "<p class='sub'>トレードはありませんでした。</p>"

    SN = {"long": "買い", "short": "売り"}
    srows = [[(SN[sd], ""), (str(st["トレード数"]), ""), (_f(st["勝率 [%]"], 1) + "%", ""),
              (_f(st["合計R"], 2, True), _cls(st["合計R"])), (_f(st["PF"]), ""), (_f(st["最大DD [R]"], 1), "")]
             for sd, st in stats_side.items()]
    side_tbl = table(["売買", "取引", "勝率", "合計R", "PF", "最大DD"], srows,
                     [("合計", ""), (str(stats["トレード数"]), ""), (_f(stats["勝率 [%]"], 1) + "%", ""),
                      (_f(stats["合計R"], 2, True), ""), (_f(stats["PF"]), ""), (_f(stats["最大DD [R]"], 1), "")])
    order = ["開始", "転換(CHoCH)", "RR接触", "エントリー"]
    keys = order + [k for k in next(iter(funnels.values())) if k not in order]
    frows = [[(escape(k), "" if k in order else "muted")] + [(str(f.get(k, 0)), "" if k in order else "muted")
                                                              for f in funnels.values()] for k in keys]
    funnel_tbl = table(["段階"] + [SN[sd] for sd in funnels], frows)

    rr_txt = f"エントリー + 損切り幅 × {cfg.rr_target}" if cfg.rr_target else "上位足の直近スイング高値"
    html = f"""<title>上位足押し目×下位足転換 {escape(cfg.pair)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>
{CSS}</style>
<main class="wrap">
<header><h1>上位足押し目 × 下位足転換 {escape(cfg.pair)}</h1>
<p class="sub">{escape(period_label)} / 上位足 {escape(cfg.htf_rule)} × 下位足 {cfg.ltf_hours}h / {"・".join(SN[x] for x in stats_side)} / TP {cfg.rr_target}R / 1分足執行 / スプレッド課金後のR</p></header>

<section><h2>結果</h2>
{kpi}
</section>

<section><h2>累積R</h2><div class="card">{equity_svg(trades["exit_time"] if len(trades) else [], r)}</div>
<p class="hint muted">決済順の累積R（スプレッド課金後）。</p></section>

<section><h2>売買別</h2>{side_tbl}
<p class="hint muted">買いと売りは別々に検証して合算しています。同じ時期に買いと売りのポジションが重なることがあります。</p></section>

<section><h2>年別</h2>{yearly}</section>

<section><h2>セットアップの段階ごとの件数</h2>{funnel_tbl}
<p class="hint muted">上位足ゾーンへの押しで「開始」し、下位足の CHoCH → RR水準への接触 → トリガーの順に進んだ数。取り消しは各段階の合計。</p></section>

<section><h2>チャート</h2>
<div style="display:grid;gap:10px">
<a class="link" href="setups.html"><b>セットアップ一覧（{len(r)}件）</b><span>下位足のローソク足に、上位足ゾーン・H_a / H_b・底・HL・接触・エントリー・SL / TP を描いた確認用の図。検出が目視の形と合っているかをここで確かめます。</span></a>
<a class="link" href="{escape(chart_file)}"><b>トレード付きダウ理論チャート</b><span>エントリー ▲・決済 ●・SL/TP。背景は表示中の時間足のダウ理論判定（約2.7MB）。エントリーラインの表示は既存のスイングブレイク用で、この戦略とは無関係です。</span></a>
</div></section>

<section><h2>トレード一覧</h2>{trade_tbl}
<p class="hint muted"># をタップするとセットアップ図に移動します。時刻は NY 時間。</p></section>

<section><h2>売買ルールと設定</h2>
<ul>
<li><b>売り:</b> 買いの条件の高値と安値を入れ替えた対称ルールです（上位足の下降トレンド、上抜けされた前回高値の代わりに下抜けされた前回安値、H_a / H_b は L_a / L_b、トリガーは直近 {cfg.trigger_n} 本の安値の下抜け）。</li>
<li><b>足:</b> 1分足から NY 17:00 区切りで下位足（{cfg.ltf_hours}時間足）を作り、上位足（{escape(cfg.htf_rule)}）は下位足をまとめて作ります。</li>
<li><b>スイング:</b> 左右 n 本のフラクタル（上位足 n={cfg.swing_n_htf}、下位足 n={cfg.swing_n_ltf}）。同値は最初の1本だけ。ピボットの n 本後から使います。</li>
<li><b>上位足:</b> 直近2つのスイング高値・安値がともに切り上がっていれば上昇。ゾーン = 上抜けされた前回スイング高値 ± 上位足ATR × {cfg.htf_zone_atr}。上位足が確定した後の下位足にだけ使います。</li>
<li><b>開始:</b> 上昇中に、下位足の安値がゾーン上限以下・終値がゾーン下限以上。</li>
<li><b>転換（CHoCH）:</b> 底より前の確定済みスイング高値の直近2つが切り下がり（H_a &gt; H_b）、終値が H_b を上抜け。底の後の高値（H_c）は使いません。否定ライン = 底の後の押し安値（無ければ底）。</li>
<li><b>RR水準への押し:</b> 高値が H_b + max(下位足ATR × {cfg.rr_away_atr}, 上位足ATR × {cfg.rr_away_htf_atr}) まで離れた後（上位足ATR側は仕様書にない追加）、安値が H_b + ATR × {cfg.rr_zone_atr} 以下で接触。割り込みの深さは制限しません。</li>
<li><b>トリガー:</b> 終値が直近 {cfg.trigger_n} 本の高値を上抜け → 次の足の始値で買い。各段階の期限 {cfg.setup_expiry} 本、終値がゾーン下限・否定ラインを割れば取り消し。</li>
<li><b>決済:</b> SL = 接触安値 − ATR × {cfg.sl_buffer_atr}、TP = {rr_txt}。SL / TP は1分足で判定し、同じ足で両方に触れたら SL 優先。ATR は EMA の {cfg.atr_len} 本。ポジションは1つだけ。</li>
<li><b>コスト:</b> 平均相対スプレッド {spread * 100:.5f}% の半分をエントリーと決済のそれぞれに課金（R から差し引き）。</li>
</ul></section>

<section><h2>読むときの注意</h2>
<ul>
<li>取引数が {len(r)} 件と少なく、勝率や PF のぶれが大きい段階です。結論より、まずセットアップ一覧で検出の形が意図どおりかを確かめる用途です。</li>
<li>パラメータは仕様書の既定値のままで、合わせ込みはしていません。期間の分割（ウォークフォワード）もまだ行っていません。</li>
<li>買いと売りの合計は、2本の独立した検証の単純合算です。同時保有の制限や資金の共有は考えていません。</li>
<li>検証開始より前の数ヶ月はスイングと ATR の助走期間で、セットアップは {escape(str(cfg.start_date))} 以降の足からだけ始めています。</li>
</ul></section>
</main>
"""
    Path(path).write_text(html, encoding="utf-8")
    return path
