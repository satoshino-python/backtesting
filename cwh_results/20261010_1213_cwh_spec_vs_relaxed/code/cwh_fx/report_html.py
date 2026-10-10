"""
cwh_results/<実行>/results.json から report.html（スマホ1カラム、ライト/ダーク対応）を作る。
見た目は filter_results/trend_filter_report.html（トレンドフィルター検証）に揃えている。

  python cwh_fx/report_html.py cwh_results/<実行フォルダ>

「わかったこと」と「注意点」は、同じフォルダに findings.json（{"verdict": "...", "verdict_sub": "...",
"findings": [{"pill": "good|bad|mid", "tag": "...", "title": "...", "body": "..."}], "notes": ["..."]}）があればその文章を使う。
"""
import html
import json
import math
import sys
from pathlib import Path

E = html.escape

CSS = r"""
/* Layout: one phone-width column — headline numbers, curve, tables, diagnosis, validation, charts, rules, caveats. Same tokens as the トレンドフィルター検証 report. */
:root {
  --bg: #f3f5f7; --surface: #ffffff; --line: #dde2e8; --fg: #17202b; --muted: #5e6b7a;
  --accent: #2a5bd7; --accent-soft: #e3eafb; --s2: #b8650a; --s3: #7d8a99;
  --pos: #18865a; --neg: #c8413a; --pos-bg: rgba(24,134,90,.14); --neg-bg: rgba(200,65,58,.14);
  --f-head: "Zen Kaku Gothic New", "Hiragino Sans", "Noto Sans JP", system-ui, sans-serif;
  --f-num: "IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #0f141a; --surface: #171e26; --line: #2a333e; --fg: #e3e8ee; --muted: #93a0ae;
  --accent: #79a2ff; --accent-soft: #1d2a45; --s2: #e9a04a; --s3: #8c99a8;
  --pos: #3ccb8b; --neg: #f0726a; --pos-bg: rgba(60,203,139,.16); --neg-bg: rgba(240,114,106,.16);
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #0f141a; --surface: #171e26; --line: #2a333e; --fg: #e3e8ee; --muted: #93a0ae;
  --accent: #79a2ff; --accent-soft: #1d2a45; --s2: #e9a04a; --s3: #8c99a8;
  --pos: #3ccb8b; --neg: #f0726a; --pos-bg: rgba(60,203,139,.16); --neg-bg: rgba(240,114,106,.16);
  color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--f-head); -webkit-text-size-adjust: 100%; }
.wrap { max-width: 860px; margin: 0 auto; padding: 20px 16px 48px; display: grid; gap: 28px; }
.wrap > * { min-width: 0; }
h1 { font-size: 1.45rem; line-height: 1.3; margin: 0; font-weight: 700; text-wrap: balance; }
h2 { font-size: 1.05rem; margin: 0 0 10px; font-weight: 700; text-wrap: balance; }
h3 { text-wrap: balance; }
p { margin: 0; }
.sub { color: var(--muted); font-size: .85rem; margin-top: 6px; }
.num { font-family: var(--f-num); font-variant-numeric: tabular-nums; }
.pos { color: var(--pos); } .neg { color: var(--neg); }
.muted { color: var(--muted); }
.verdict { background: var(--surface); border: 1.5px solid var(--accent); border-radius: 14px; padding: 16px; display: grid; gap: 14px; }
.verdict .tag { font-size: .72rem; color: var(--accent); letter-spacing: .06em; font-weight: 500; }
.verdict .rule { font-size: 1.08rem; font-weight: 700; line-height: 1.5; text-wrap: balance; }
.findings { display: grid; gap: 10px; }
.find { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px; display: grid; gap: 4px; }
.find h3 { margin: 0; font-size: .95rem; display: flex; gap: 8px; align-items: baseline; }
.find .pill, .pill { font-size: .68rem; font-weight: 500; padding: 1px 8px; border-radius: 999px; white-space: nowrap; }
.pill.good { background: var(--pos-bg); color: var(--pos); }
.pill.bad { background: var(--neg-bg); color: var(--neg); }
.pill.mid { background: var(--accent-soft); color: var(--accent); }
.find p { font-size: .88rem; color: var(--muted); }
.find p b { color: var(--fg); font-weight: 500; }
.scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }
table { border-collapse: collapse; width: 100%; font-size: .8rem; }
th, td { padding: 7px 9px; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }
th { font-weight: 500; color: var(--muted); font-size: .72rem; background: var(--surface); }
th:first-child, td:first-child { text-align: left; position: sticky; left: 0; background: var(--surface); font-weight: 500; }
td:first-child { white-space: normal; min-width: 130px; max-width: 200px; font-size: .78rem; line-height: 1.4; }
td.l, th.l { text-align: left; }
tr:last-child td { border-bottom: 0; }
tr.base td { background: var(--accent-soft); }
td.cell { font-family: var(--f-num); font-variant-numeric: tabular-nums; }
td.hp { background: var(--pos-bg); } td.hn { background: var(--neg-bg); }
td.zero { color: var(--muted); }
.hint { font-size: .8rem; color: var(--muted); margin-top: 8px; }
.defs { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; }
.def { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px; min-width: 0; }
.def h3 { margin: 0 0 6px; font-size: .9rem; }
.def p { font-size: .86rem; }
.def code, .hint code, li code { font-family: var(--f-num); font-size: .78rem; background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: .76rem; color: var(--muted); margin-bottom: 8px; }
.legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.plot { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 10px 8px 4px; position: relative; }
.plot svg { width: 100%; height: auto; display: block; }
.plot svg text { font-family: var(--f-num); font-size: 10px; fill: var(--muted); }
.plot .grid { stroke: var(--line); stroke-width: 1; }
.plot .zero { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
.tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font: 12px/1.45 var(--f-num); padding: 6px 8px; border-radius: 6px; white-space: pre; transform: translate(-50%, -110%); }
.links { display: grid; gap: 8px; }
.links a { display: flex; justify-content: space-between; gap: 10px; align-items: center; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; color: var(--fg); text-decoration: none; }
.links a:hover, .links a:focus-visible { border-color: var(--accent); outline: none; }
.links a span { color: var(--muted); font-size: .78rem; font-family: var(--f-num); }
.hbar { display: grid; grid-template-columns: minmax(0, 8.5rem) 1fr 3.2rem; gap: 8px; align-items: center; font-size: .78rem; }
.hbar .track { height: 10px; background: var(--accent-soft); border-radius: 3px; overflow: hidden; }
.hbar .track b { display: block; height: 100%; background: var(--accent); border-radius: 0 3px 3px 0; }
.hbars { display: grid; gap: 5px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }
ul.notes { margin: 0; padding-left: 1.2em; display: grid; gap: 8px; font-size: .88rem; }
footer { font-size: .75rem; color: var(--muted); line-height: 1.7; }
footer code { font-family: var(--f-num); font-size: .72rem; word-break: break-all; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

SERIES = ["var(--accent)", "var(--s2)", "var(--s3)"]


def f(x, nd=2, sign=False, pct=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    if pct:
        return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def rcell(x, nd=2):
    if x is None:
        return '<td class="cell zero">–</td>'
    cls = "hp" if x > 0.005 else "hn" if x < -0.005 else "zero"
    return f'<td class="cell {cls}">{f(x, nd, sign=True)}</td>'


def table(head, rows, base_first=False):
    h = "".join(f"<th>{E(c)}</th>" for c in head)
    body = []
    for k, r in enumerate(rows):
        cls = ' class="base"' if base_first and k == 0 else ""
        body.append(f"<tr{cls}>" + "".join(r) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def td(x, cls="cell"):
    return f'<td class="{cls}">{x}</td>'


def curve_svg(variants):
    """決済日順の累積R（ステップ）。トレードの無い設定は 0 の線"""
    series = []
    for k, v in enumerate(variants):
        tr = sorted(v["trades"], key=lambda t: t["exit_date"])
        pts, cum = [], 0.0
        for t in tr:
            cum += t["R"]
            pts.append((str(t["exit_date"])[:10], cum, t))
        series.append((v, pts, SERIES[k % len(SERIES)]))
    all_dates = [p[0] for _, pts, _ in series for p in pts]
    if not all_dates:
        return "<p class='hint'>取引がないため、損益曲線はありません。</p>"
    import datetime as dt
    d0 = dt.date.fromisoformat(min([variants[0]["equity"][0][0][:10]] + all_dates))
    d1 = dt.date.fromisoformat(max([variants[0]["equity"][-1][0][:10]] + all_dates))
    span = max((d1 - d0).days, 1)
    vals = [0.0] + [p[1] for _, pts, _ in series for p in pts]
    lo, hi = math.floor(min(vals)) - 1, math.ceil(max(vals)) + 1
    W, H, L, R, T, B = 640, 260, 34, 10, 10, 24
    X = lambda d: L + (dt.date.fromisoformat(d) - d0).days / span * (W - L - R)
    Y = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
    g = []
    step = max(1, int(math.ceil((hi - lo) / 6)))
    v = math.ceil(lo / step) * step
    while v <= hi:
        g.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                 f'<text x="{L - 6}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:+d}</text>')
        v += step
    g.append(f'<line class="zero" x1="{L}" x2="{W - R}" y1="{Y(0):.1f}" y2="{Y(0):.1f}"/>')
    for y in range(d0.year, d1.year + 1):
        x = X(f"{y}-01-01") if dt.date(y, 1, 1) >= d0 else None
        if x is not None:
            g.append(f'<text x="{x:.1f}" y="{H - 6}" text-anchor="middle">{y}</text>')
    lines, dots = [], []
    for v, pts, col in series:
        if not pts:
            continue
        path = f"M{X(str(d0)):.1f},{Y(0):.1f}"
        prev = 0.0
        for d, c, t in pts:
            path += f" H{X(d):.1f} V{Y(c):.1f}"
            prev = c
        path += f" H{X(str(d1)):.1f}"
        lines.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"/>')
        for d, c, t in pts:
            tip = (f"{v['name']}\n{t['pair']} {t['side']} {str(t['entry_date'])[:10]}→{d}\n"
                   f"{t['exit_reason']}  {t['R']:+.2f}R  累計 {c:+.2f}R")
            dots.append(f'<circle cx="{X(d):.1f}" cy="{Y(c):.1f}" r="3.5" fill="{col}" stroke="var(--surface)" '
                        f'stroke-width="2" data-tip="{E(tip)}"/>'
                        f'<circle cx="{X(d):.1f}" cy="{Y(c):.1f}" r="11" fill="transparent" data-tip="{E(tip)}"/>')
    legend = "".join(f'<span><i style="background:{col}"></i>{E(v["name"])}（{len(pts)}件）</span>'
                     for v, pts, col in series)
    return (f'<div class="legend">{legend}</div><div class="plot" id="curve">'
            f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="累積R">{"".join(g)}{"".join(lines)}{"".join(dots)}</svg>'
            f'<div class="tip" id="tip" hidden></div></div>')


def build_report(out):
    out = Path(out)
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    fj = out / "findings.json"
    fd = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else {}
    base, vs = res["base"], res["variants"]
    pairs = sorted({r["pair"] for r in res["by_pair"]})
    keys = [v["key"] for v in vs]
    name = {v["key"]: v["name"] for v in vs}
    acct = base["account_currency"]
    risk_amt = base["initial_equity"] * base["risk_per_trade"]

    # 1. 主な数字
    heads = ["設定", "取引数", "合計R", "PF", "最大DD(R)", "最大DD(資産)", "勝率", "平均R"]
    rows = []
    for v in vs:
        m = v["metrics"]
        rows.append([td(E(v["name"]), ""), td(m.get("trades", 0)), rcell(m.get("total_r")),
                     td(f(m.get("pf"))), td(f(m.get("max_dd_r"))), td(f(m.get("max_dd_pct"), pct=True)),
                     td(f(m.get("win_rate"), pct=True)), rcell(m.get("avg_r"), 3)])
    key_table = table(heads, rows, base_first=True)

    # 成績一覧（全指標）
    heads2 = ["設定", "シグナル", "約定", "見送り", "買い/売り", "年率", "シャープ", "ソルティノ", "最大連敗",
              "平均勝ち", "平均負け", "保有(中央値)", "最終資産"]
    rows2 = []
    for v in vs:
        m = v["metrics"]
        rows2.append([td(E(v["name"]), ""), td(m.get("signals", 0)), td(m.get("trades", 0)), td(m.get("skipped", 0)),
                      td(f'{m.get("long", 0)}/{m.get("short", 0)}'), td(f(m.get("cagr"), pct=True, sign=True)),
                      td(f(m.get("sharpe"))), td(f(m.get("sortino"))), td(m.get("max_losing_streak", 0)),
                      td(f(m.get("avg_win_r"), sign=True)), td(f(m.get("avg_loss_r"), sign=True)),
                      td(f(m.get("median_bars"), 0) + "本" if m.get("median_bars") is not None else "–"),
                      td(f'{m.get("final_equity", 0):,.0f}')])
    full_table = table(heads2, rows2, base_first=True)

    # 年別・ペア別
    years = sorted({r["year"] for r in res["yearly"]})
    yr = {(r["key"], r["year"]): r for r in res["yearly"]}
    year_rows = [[td(str(y), "")] + [rcell(yr[(k, y)]["total_r"]) if (k, y) in yr else '<td class="cell zero">0件</td>'
                                     for k in keys] for y in years]
    year_table = table(["決済した年"] + [name[k] for k in keys], year_rows) if years else "<p class='hint'>取引なし</p>"
    bp = {(r["key"], r["pair"]): r for r in res["by_pair"]}
    pair_rows = []
    for p in pairs:
        row = [td(p, "")]
        for k in keys:
            r = bp[(k, p)]
            row.append(rcell(r["total_r"]) if r["trades"] else '<td class="cell zero">0件</td>')
            row.append(td(r["trades"]))
        pair_rows.append(row)
    pair_head = ["ペア"] + [x for k in keys for x in (f"{name[k]} R", "件数")]
    pair_table = table(pair_head, pair_rows)

    # 取引が少ない理由
    fun = res["funnel"]
    k0 = keys[0]
    stages = []
    for r in fun:
        if r["key"] == k0 and r["stage"] not in stages:
            stages.append(r["stage"])
    fv = {(r["key"], r["direction"], r["stage"]): r["bars"] for r in fun}
    tot = {d: sum(fv.get((k0, d, s), 0) for s in stages) for d in (1, -1)}
    fun_rows = []
    for s in stages:
        a, b = fv.get((k0, 1, s), 0), fv.get((k0, -1, s), 0)
        if a == 0 and b == 0 and s not in ("成立",):
            continue
        fun_rows.append([td(E(s), ""), td(f"{a:,}"), td(f"{b:,}")])
    fun_table = table(["いちばん先まで進んだ候補が落ちた条件", "買い（足数）", "売り（足数）"], fun_rows)
    loo = res["loo"]
    rel = []
    for r in loo:
        if r["relaxed"] not in rel:
            rel.append(r["relaxed"])
    lv = {(r["key"], r["relaxed"]): r["signals"] for r in loo}
    loo_rows = [[td(E(x), "")] + [td(lv.get((k, x), "–")) for k in keys] for x in rel]
    loo_table = table(["外した条件"] + [name[k] for k in keys], loo_rows, base_first=True)

    # 見送り
    skip_rows = []
    for v in vs:
        cnt = {}
        for s in v["skipped"]:
            reason = "損切り幅が 1〜3×ATR14 の範囲外" if s["reason"].startswith("損切り幅") else s["reason"]
            cnt[reason] = cnt.get(reason, 0) + 1
        for reason, n in sorted(cnt.items(), key=lambda x: -x[1]):
            skip_rows.append([td(E(v["name"]), ""), td(E(reason), "l"), td(n)])
    skip_table = table(["設定", "見送りの理由", "件数"], skip_rows) if skip_rows else "<p class='hint'>見送りなし</p>"

    # 取引一覧
    trade_blocks = []
    for v in vs:
        if not v["trades"]:
            continue
        rows_t = []
        for t in sorted(v["trades"], key=lambda t: t["entry_date"]):
            rows_t.append([td(f'{t["pair"]} {t["side"]}', ""), td(str(t["entry_date"])[:16]), td(str(t["exit_date"])[:16]),
                           td(E(t["exit_reason"]), "l"), rcell(t["R"]), td(f'{t["bars_held"]}本'),
                           td(f'{t["depth_atr"]:.1f}'), td(f'{t["cup_len"]}/{t["handle_len"]}'), td(f'{t["stop_atr"]:.2f}'),
                           td(f'{t["lots"]:.2f}')])
        trade_blocks.append(f"<h3 style='font-size:.9rem;margin:14px 0 8px'>{E(v['name'])}</h3>" + table(
            ["ペア", "エントリー", "決済", "決済理由", "R", "保有", "深さATR", "カップ/ハンドル本数", "損切りATR14", "ロット"], rows_t))

    # 検証
    val_html = ""
    va = res.get("validation")
    if va:
        vname = name[va["key"]]
        def mrow(r):
            return [td(E(r["name"]), ""), td(r["trades"]), rcell(r["total_r"]), rcell(r["avg_r"], 3),
                    td(f(r["win_rate"], pct=True)), td(f(r["pf"])), td(f(r["max_dd_r"]))]
        mh = ["", "取引数", "合計R", "平均R", "勝率", "PF", "最大DD(R)"]
        cost_t = table(mh, [mrow(r) for r in va["cost"]], base_first=True)
        param_t = table(mh, [mrow(r) for r in va["param"]], base_first=True)
        wf_rows = [[td(f'{r["train"]} → {r["test"]}', ""), td(E(r["pick"]), "l"), td(r["train_trades"]), rcell(r["train_r"]),
                    td(r["test_trades"]), rcell(r["test_r"]), td(r["base_test_trades"]), rcell(r["base_test_r"])]
                   for r in va["walk_forward"]]
        wf_t = table(["学習 → 検証", "学習期間で選んだ設定", "学習 件数", "学習 R", "検証 件数", "検証 R", "初期値 件数", "初期値 R"], wf_rows)
        ps_rows = [[td(E(r["train_pairs"]), ""), td(E(r["pick"]), "l"), td(r["train_trades"]), rcell(r["train_r"]),
                    td(E(r["test_pairs"]), "l"), td(r["test_trades"]), rcell(r["test_r"]), td(r["base_test_trades"]), rcell(r["base_test_r"])]
                   for r in va["pair_split"]]
        ps_t = table(["選んだペア", "選んだ設定", "件数", "R", "確かめたペア", "件数", "R", "初期値 件数", "初期値 R"], ps_rows)
        rnd_rows = []
        for r in va["random"]:
            rnd_rows.append([td("SMA200 の向きに合わせたランダム" if r["trend_aligned"] else "完全なランダム", ""),
                             td(r["n_trades"]), rcell(r["actual_avg_r"], 3), rcell(r["rand_mean"], 3),
                             td(f'{r["rand_p5"]:+.3f} 〜 {r["rand_p95"]:+.3f}'), td(f'{r["p_value"]:.2f}')])
        rnd_t = table(["比較の相手", "取引数", "実際の平均R", "ランダムの平均R", "ランダムの5〜95%", "ランダムが上回る割合"], rnd_rows)
        val_html = f"""
  <section>
    <h2>検証（仕様12章）: {E(vname)}</h2>
    <p class="hint" style="margin:0 0 10px">取引数が少ないため、どの数字も偶然で大きく動く。傾向を見る程度にとどめる。</p>
    <h3 style="font-size:.9rem;margin:0 0 8px">コスト感度（12.3）</h3>{cost_t}
    <h3 style="font-size:.9rem;margin:16px 0 8px">パラメータ感度 ±20%（12.4）</h3>{param_t}
    <p class="hint">深さの範囲は下限・上限を同じ倍率で、ハンドル長は 5〜25本 の両端を同じ倍率で動かした。</p>
    <h3 style="font-size:.9rem;margin:16px 0 8px">ウォークフォワード 学習3年 → 検証1年（12.5）</h3>{wf_t}
    <p class="hint">深さ・trail_atr_mult・ハンドル長をそれぞれ ×0.8 / 1.0 / 1.2 の27通りから、学習期間の合計Rが最大のものを選んで次の1年に使った。検証期間を合わせた成績: {va["walk_forward_oos"]["trades"]}件 / {va["walk_forward_oos"]["total_r"]:+.2f}R。</p>
    <h3 style="font-size:.9rem;margin:16px 0 8px">銘柄分割（12.7）</h3>{ps_t}
    <h3 style="font-size:.9rem;margin:16px 0 8px">ランダムエントリーとの比較（12.6）</h3>{rnd_t}
    <p class="hint">同じ回数・同じ決済ルール（2Rで半分利確→建値、3×ATR14 のチャンデリア、40本のタイムストップ）・同じ損切り幅の分布で {va["random"][0]["n_sims"] if va["random"] else 0} 回くり返した。「ランダムが上回る割合」が小さいほど、パターンに意味がある。</p>
  </section>"""

    # チャート
    ch = res.get("charts", [])
    chart_links = "".join(f'<a href="{E(c["file"])}"><b>{E(c["pair"])} ・ {E(name[c["key"]])}</b>'
                          f'<span>{c["trades"]}件 / {c["mb"]}MB</span></a>' for c in ch)
    chart_html = (f'<div class="links">{chart_links}</div><p class="hint">開くとダウ理論チャート（1H/4H/D1/W1）にエントリー▲▼・決済●・損切り/部分利確のラインを表示する。'
                  f'背景は表示中の時間足で計算したダウ理論のトレンド。エントリーラインの代わりに、左縁の候補になるスイング（左右{base.get("swing_window", 5)}本）を、移動平均は SMA50/200 を表示している。</p>'
                  if ch else "<p class='hint'>取引のある設定がないため、チャートはありません。</p>")

    # 「わかったこと」
    verdict = fd.get("verdict", "")
    verdict_sub = fd.get("verdict_sub", "")
    finds = "".join(f'<div class="find"><h3><span class="pill {E(x.get("pill", "mid"))}">{E(x.get("tag", ""))}</span>{E(x["title"])}</h3>'
                    f'<p>{x["body"]}</p></div>' for x in fd.get("findings", []))
    notes = "".join(f"<li>{n}</li>" for n in fd.get("notes", []))
    over_txt = "".join(f"<li><b>{E(v['name'])}</b>: " + (", ".join(f"<code>{E(k)}={E(str(x))}</code>" for k, x in v["overrides"].items()) or "変更なし（config.yaml の初期値）") + "</li>" for v in vs)
    sp = base.get("spread_pips", {})
    sp_txt = "、".join(f"{p} {sp[p]}" for p in pairs if p in sp)

    title = fd.get("title", "カップウィズハンドル FX 5ペア")
    page = f"""<title>{E(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Zen+Kaku+Gothic+New:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header>
    <h1>{E(title)}</h1>
    <p class="sub">ATRベースの仕様書どおりに実装 / {", ".join(pairs)} / 取引日 {base["start_date"]} 〜 {base["end_date"]} / GMOクリック証券の1分足（BID）から足を作成 / 実行 {res["created"]}</p>
  </header>

  <section class="verdict">
    <span class="tag">結論</span>
    <p class="rule">{verdict}</p>
    {key_table}
    <p class="sub" style="margin:0">{verdict_sub}</p>
  </section>

  {"<section><h2>わかったこと</h2><div class='findings'>" + finds + "</div></section>" if finds else ""}

  <section>
    <h2>損益曲線（決済順の累積R）</h2>
    {curve_svg(vs)}
    <p class="hint">1R = その取引の初期リスク（約定価格〜初期損切り × 枚数）。部分利確とスワップ・コストを含む。点に触れると取引の内容を表示。</p>
  </section>

  <section>
    <h2>年別の合計R</h2>
    {year_table}
  </section>

  <section>
    <h2>ペア別の合計R</h2>
    {pair_table}
  </section>

  <section>
    <h2>成績一覧</h2>
    {full_table}
    <p class="hint">年率・シャープ・ソルティノは日々の資産（含み損益込み、{acct}、初期 {base["initial_equity"]:,.0f}、1回のリスク {base["risk_per_trade"] * 100:.1f}%）から計算。最大DD(R) は決済順の累積Rの最大下落幅。</p>
  </section>

  <section>
    <h2>取引が少ない理由</h2>
    <h3 style="font-size:.9rem;margin:0 0 8px">条件を1つだけ外したときのシグナル数</h3>
    {loo_table}
    <p class="hint">青い行がすべての条件を満たしたシグナル数（ポジション上限などで見送る前）。行ごとに、その条件だけを必ず通るようにして数え直した。</p>
    <h3 style="font-size:.9rem;margin:16px 0 8px">どこで落ちたか（{E(name[k0])}、全ペアの足数）</h3>
    {fun_table}
    <p class="hint">足ごとに、左縁の候補のうち一番先の条件まで進んだものが、どの条件で落ちたかを数えた（判定の順は表の上から）。買い {tot[1]:,} 本・売り {tot[-1]:,} 本。</p>
  </section>

  <section>
    <h2>見送ったシグナル</h2>
    {skip_table}
  </section>

  <section>
    <h2>取引一覧</h2>
    {"".join(trade_blocks) or "<p class='hint'>取引なし</p>"}
  </section>
{val_html}
  <section>
    <h2>トレード付きのダウ理論チャート</h2>
    {chart_html}
  </section>

  <section>
    <h2>売買ルール</h2>
    <div class="defs">
      <div class="def"><h3>トレンド</h3><p>終値 &gt; SMA200 かつ SMA200 が20本前より上（売りは反対）。</p></div>
      <div class="def"><h3>左縁とカップ</h3><p>左右5本で最高値のスイングハイが左縁。その前60本の安値から <code>6×ATR50</code> 以上の上昇。左縁から右縁まで30〜150本、深さ <code>3〜10×ATR50</code>、底は期間の25〜75%の位置、底から <code>1.5×ATR50</code> 以内の安値の足が20%以上、右縁は左縁高値 <code>−1×ATR50</code> 以上まで回復。</p></div>
      <div class="def"><h3>ハンドル</h3><p>右縁の翌足から5〜25本。押しは <code>0.5〜2.5×ATR50</code> かつ深さの半分以下、カップの上半分。ハンドル中の ATR14 の平均 &lt; <code>0.8×ATR50</code>。</p></div>
      <div class="def"><h3>エントリー</h3><p>終値が右縁高値 + <code>0.1×ATR14</code> を超えたら、次の足の始値で成行。ハンドル完成から10本でブレイクしなければ失効、終値がハンドル安値を割ったら破棄。始値がピボットから <code>1×ATR14</code> 超離れていたら見送り。</p></div>
      <div class="def"><h3>決済</h3><p>初期損切り = ハンドル安値 − <code>0.5×ATR14</code>（幅が <code>1〜3×ATR14</code> でなければ見送り）。+2Rで半分を利確し、残りの損切りを建値へ。チャンデリア（エントリー後の高値 − <code>3×ATR14</code>）を終値で割ったら次の始値で決済。40本で+1R未満なら次の始値で決済。</p></div>
      <div class="def"><h3>資金管理とコスト</h3><p>口座 {acct}、1回のリスクは資産の {base["risk_per_trade"] * 100:.1f}%（0.01ロット単位で切り捨て、レバレッジ{base["max_leverage"]:.0f}倍まで）。同時保有は最大{base["max_positions"]}、同じ通貨を含むのは最大{base["max_exposure_per_currency"]}。スプレッド（pips）: {sp_txt}。スリッページ 成行{base["slippage_pips"]} / 逆指値{base["stop_slippage_pips"]}pips。スワップは未設定（0）。</p></div>
    </div>
    <ul class="notes" style="margin-top:12px">{over_txt}</ul>
  </section>

  {"<section><h2>注意点</h2><ul class='notes'>" + notes + "</ul></section>" if notes else ""}

  <footer>
    コード: <code>cwh_fx/</code>（<code>src/pattern.py</code> 検出、<code>src/signals.py</code> シグナル、<code>src/backtest.py</code> イベント駆動バックテスト、<code>src/validation.py</code> 12章の検証、<code>run_backtest.py</code> 実行）。テスト: <code>python -m pytest cwh_fx/tests</code>。
    結果: <code>{E(str(out))}</code>
  </footer>
</div>
<script>
(() => {{
  const plot = document.getElementById("curve"), tip = document.getElementById("tip");
  if (!plot) return;
  const show = (e) => {{
    const t = e.target.closest("[data-tip]");
    if (!t) {{ tip.hidden = true; return; }}
    const r = plot.getBoundingClientRect(), c = t.getBoundingClientRect();
    tip.textContent = t.dataset.tip; tip.hidden = false;
    const x = Math.min(Math.max(c.left + c.width / 2 - r.left, 110), r.width - 110);
    tip.style.left = x + "px"; tip.style.top = (c.top - r.top) + "px";
  }};
  plot.addEventListener("pointermove", show);
  plot.addEventListener("pointerdown", show);
  plot.addEventListener("pointerleave", () => tip.hidden = true);
}})();
</script>
"""
    (out / "report.html").write_text(page, encoding="utf-8")
    return out / "report.html"


if __name__ == "__main__":
    print(build_report(sys.argv[1]))
