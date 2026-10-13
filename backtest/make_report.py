"""run_pullback.py の出力フォルダから report.html を作る。使い方: python backtest/make_report.py <出力フォルダ> [--swing-n N] [--base <比較する取引CSV>]"""
import sys, html, pathlib
import pandas as pd

CH = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "XAUUSD"]
dec = lambda p: 2 if p == "XAUUSD" else 3 if p.endswith("JPY") else 5
pf = lambda s: s[s > 0].sum() / -s[s < 0].sum()


def main(D, swing_n=None, base=None):
    t = pd.read_csv(D / "rule_e_trades.csv")
    g = t.groupby("pair").agg(n=("R", "size"), win=("R", lambda s: (s > 0).mean()), sumR=("R", "sum"),
                              avgR=("R", "mean"), days=("days", "mean")).reset_index()
    g["pf"] = [pf(t[t.pair == p].R) for p in g.pair]
    rows = "".join(f"<tr><td>{r.pair}</td><td>{r.n}</td><td>{r.win:.0%}</td><td class='{'pos' if r.sumR > 0 else 'neg'}'>{r.sumR:+.1f}</td><td>{r.avgR:+.2f}</td><td>{r.pf:.2f}</td><td>{r.days:.0f}</td><td>{'<a href=\"chart_' + r.pair + '.html\">チャート</a>' if r.pair in CH else ''}</td></tr>" for r in g.itertuples())
    t["year"] = t.entry_date.str[:4]
    y = t.groupby("year").R.agg(["size", "sum"]).reset_index()
    yrows = "".join(f"<tr><td>{r.year}</td><td>{r['size']}</td><td class='{'pos' if r['sum'] > 0 else 'neg'}'>{r['sum']:+.1f}</td></tr>" for _, r in y.iterrows())
    rs = t.reason.value_counts()
    tot, n, wr, PF = t.R.sum(), len(t), (t.R > 0).mean(), pf(t.R)
    e = t.sort_values("exit_date").R.cumsum().values
    W, H = 640, 200; mx, mn = max(e.max(), 0), min(e.min(), 0)
    pts = " ".join(f"{i / (len(e) - 1) * W:.1f},{H - (v - mn) / (mx - mn) * H:.1f}" for i, v in enumerate(e)); z = H - (0 - mn) / (mx - mn) * H
    svg = (f'<svg viewBox="-4 -8 {W + 8} {H + 16}" role="img" aria-label="累積R"><line x1="0" x2="{W}" y1="{z:.1f}" y2="{z:.1f}" stroke="var(--line)"/>'
           f'<polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="2"/><text x="0" y="-1" font-size="11" fill="var(--mute)">最大 {mx:+.0f}R</text>'
           f'<text x="0" y="{H + 12}" font-size="11" fill="var(--mute)">最小 {mn:+.0f}R / 終値 {e[-1]:+.1f}R（全銘柄の取引を決済順に累積）</text></svg>')
    tabs = ""
    for p in CH:
        s = t[t.pair == p]; d = dec(p)
        tr = "".join(f"<tr><td>{'買' if r.dir == 1 else '売'}</td><td>{r.L0_date}</td><td>{r.H_date}</td><td>{r.D_date}</td><td>{r.entry_date}</td><td>{r.entry_price:.{d}f}</td><td>{r.exit_date}</td><td>{r.exit_price:.{d}f}</td><td>{r.L0:.{d}f}</td><td>{r.H:.{d}f}</td><td>{r.D:.{d}f}</td><td>{r.retr:.0%}</td><td>{r.days}</td><td>{html.escape(r.reason)}</td><td class='{'pos' if r.R > 0 else 'neg'}'>{r.R:+.2f}</td></tr>" for r in s.itertuples())
        tabs += f"<details><summary>{p}（{len(s)}件 / {s.R.sum():+.1f}R）<a href='chart_{p}.html'>チャートを開く</a></summary><div class='scroll'><table><thead><tr><th>向き</th><th>L0日</th><th>H日</th><th>D日</th><th>エントリー日</th><th>価格</th><th>決済日</th><th>価格</th><th>L0</th><th>H</th><th>D</th><th>押し</th><th>日数</th><th>決済理由</th><th>R</th></tr></thead><tbody>{tr}</tbody></table></div></details>"
    if swing_n:
        rule = (f"押し安値 L0 は<b>日足の終値で前後{swing_n}本の最安値の足（スイングロー）</b>。{swing_n}本後に確定し、確定前は使いません。"
                "H は L0 以降の最高値、D は H 以降の最安値。終値が L0 を割ったら数え直します。ATRの2倍戻りによる確定は使いません。")
        sub = f"日足・スイングロー N={swing_n} で波を定義・前回上昇幅の30〜50%押し・反発確認でエントリー。2020-2025、{t.pair.nunique()}銘柄。"
        title = f"押し目買い ルールE スイングN{swing_n}"
    else:
        rule, title = "L0 はATRの2倍戻りで確定。", "押し目買い ルールE 検証"
        sub = "日足・前回上昇幅の30〜50%押し・反発確認でエントリー。"
    cmp_html = ""
    if base:
        b = pd.read_csv(base)
        cmp_html = (f"<section><h2>ATR版との比較（同じ全銘柄）</h2><div class='scroll'><table><thead><tr><th></th><th>取引</th><th>勝率</th><th>合計R</th><th>PF</th></tr></thead><tbody>"
                    f"<tr><td>ATR2倍戻り（前回）</td><td>{len(b)}</td><td>{(b.R > 0).mean():.0%}</td><td>{b.R.sum():+.1f}</td><td>{pf(b.R):.2f}</td></tr>"
                    f"<tr><td>スイングN={swing_n}（今回）</td><td>{n}</td><td>{wr:.0%}</td><td>{tot:+.1f}</td><td>{PF:.2f}</td></tr></tbody></table></div></section>")
    page = f'''<title>{title}</title>
<style>
:root{{--bg:#f6f7f9;--fg:#1b2330;--mute:#5f6b7a;--card:#fff;--line:#d9dee5;--accent:#1f6fb2;--pos:#16794c;--neg:#b3332b;--warn:#fff4d6;--warnfg:#6b4e00}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#12161c;--fg:#e4e8ee;--mute:#9aa6b5;--card:#1a2029;--line:#2b3441;--accent:#5aa9e6;--pos:#4cc38a;--neg:#ff7b72;--warn:#3a3016;--warnfg:#f0d68a;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#12161c;--fg:#e4e8ee;--mute:#9aa6b5;--card:#1a2029;--line:#2b3441;--accent:#5aa9e6;--pos:#4cc38a;--neg:#ff7b72;--warn:#3a3016;--warnfg:#f0d68a;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font-family:"Noto Sans JP",system-ui,sans-serif;padding-inline:16px;padding-block:20px;line-height:1.6}}
main{{max-width:760px;margin:0 auto;display:flex;flex-direction:column;gap:20px;min-width:0}}
h1{{font-size:1.4rem;margin:0;text-wrap:balance}}h2{{font-size:1.05rem;margin:0 0 8px}}
.sub{{color:var(--mute);margin:4px 0 0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.kpi b{{display:block;font-size:1.35rem;font-variant-numeric:tabular-nums}}.kpi span{{color:var(--mute);font-size:.8rem}}
section{{min-width:0}}
.scroll{{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}}
table{{border-collapse:collapse;width:100%;font-size:.82rem;font-variant-numeric:tabular-nums;white-space:nowrap}}
th,td{{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line)}}th:first-child,td:first-child{{text-align:left}}
.pos{{color:var(--pos)}}.neg{{color:var(--neg)}}a{{color:var(--accent)}}
.note{{background:var(--warn);color:var(--warnfg);border-radius:8px;padding:10px 12px;font-size:.88rem}}
.note ul{{margin:4px 0 0;padding-left:1.2em}}
details{{margin-bottom:8px}}summary{{cursor:pointer;padding:6px 0;display:flex;flex-wrap:wrap;gap:6px 14px}}
svg{{width:100%;height:auto;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px}}
</style>
<main>
<header><h1>{title}</h1><p class="sub">{sub}</p></header>
<div class="kpis"><div class="kpi"><b class="{'pos' if tot > 0 else 'neg'}">{tot:+.1f}R</b><span>合計R（{t.pair.nunique()}銘柄）</span></div><div class="kpi"><b>{PF:.2f}</b><span>PF</span></div><div class="kpi"><b>{n}</b><span>取引数（買{(t.dir == 1).sum()} / 売{(t.dir == -1).sum()}）</span></div><div class="kpi"><b>{wr:.0%}</b><span>勝率</span></div></div>
{cmp_html}
<section><h2>波の定義</h2><p>{rule}</p></section>
<section><h2>チャートで確認する</h2><p>日足(D1)表示でトレードを確認できます。▲買い・▼売り、●決済、損切り位置(D)、背景は日足のトレンド判定です。黄色の折れ線が一波で、L0 → H → D → エントリーの順に結び、各点にラベルを付けています。上部の「一波（L0→H→D）」で表示を切り替えられます。</p>
<ul>{"".join(f"<li><a href='chart_{p}.html'>{p}</a></li>" for p in CH)}</ul></section>
<section><h2>累積R</h2>{svg}</section>
<section><h2>ペア別</h2><div class="scroll"><table><thead><tr><th>ペア</th><th>取引</th><th>勝率</th><th>合計R</th><th>平均R</th><th>PF</th><th>平均保有日</th><th></th></tr></thead><tbody>{rows}</tbody></table></div></section>
<section><h2>年別（エントリー年）</h2><div class="scroll"><table><thead><tr><th>年</th><th>取引</th><th>合計R</th></tr></thead><tbody>{yrows}</tbody></table></div></section>
<section><h2>決済理由</h2><p>損切り(終値&lt;D) {rs.get("損切り(終値<D)", 0)}件 / トレンド崩れ {rs.get("トレンド崩れ", 0)}件 / 最大保有日数 {rs.get("最大保有日数", 0)}件 / データ末尾 {rs.get("データ末尾", 0)}件</p></section>
<section><h2>トレード一覧（チャート対象6ペア）</h2>{tabs}<p class="sub">全銘柄の取引は rule_e_trades.csv に保存済みです。</p></section>
<section class="note"><b>注意</b><ul>
<li>スイングローの判定は終値ベース、前後{swing_n or '-'}本で同値も含む、{swing_n or '-'}本後に確定、という私の解釈です。高値・安値ベースにする場合は変更します。</li>
<li>比較ルール、感応度、2009年前後の区分は未実装です。データは2020-2025のみです。ATRは終値差のEMAです。損益はR（損切り幅=1R）で、コスト・スワップは未反映です。</li>
<li>損切りは終値がDを下回った日の終値で決済のため、-1Rを超える損失があります（最大 {t.R.min():.1f}R）。パラメータは調整していません。</li></ul></section>
</main>'''
    (D / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    a = sys.argv[1:]
    D = pathlib.Path(a[0])
    sn = int(a[a.index("--swing-n") + 1]) if "--swing-n" in a else None
    base = a[a.index("--base") + 1] if "--base" in a else None
    main(D, sn, base)
