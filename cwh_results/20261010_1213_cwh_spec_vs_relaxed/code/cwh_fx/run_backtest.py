"""
カップウィズハンドル FX のバックテストを実行する。リポジトリのルートで実行すること。

  python cwh_fx/run_backtest.py                                   # config.yaml の設定（仕様どおり）だけ
  python cwh_fx/run_backtest.py --set bar_hours=4 --label h4       # 設定を一時的に上書き（config.yaml は書き換えない）
  python cwh_fx/run_backtest.py \
      --variant spec "仕様どおり（日足）" "" \
      --variant d1_nocontr "日足・ATR収縮なし" "handle_atr_contraction=99" \
      --validate d1_nocontr                                        # 複数の設定を並べて比較し、1つに仕様12章の検証をかける

結果は cwh_results/<日時>_<ラベル>/ に保存し、前の結果は上書きしない:
  summary.csv（設定ごとの成績）、by_pair.csv、yearly.csv、trades_<key>.csv、skipped_<key>.csv、equity_<key>.csv、
  funnel.csv（どの条件で落ちたか）、leave_one_out.csv（条件を1つ外したときのシグナル数）、validation_*.csv、
  dow_trade_chart_<ペア>_<key>.html（トレード付きのダウ理論チャート）、report.html、results.json、config.json、code/
"""
import argparse
import json
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from cwh_fx.src import diagnostics, validation
from cwh_fx.src.backtest import run_portfolio
from cwh_fx.src.data import available_fx_pairs, load_1min, load_bars, pip_size_for, price_decimals_for
from cwh_fx.src.report import all_metrics, by_group, holding_distribution

CONFIG = Path(__file__).resolve().parent / "config.yaml"
RESULTS_ROOT = Path("cwh_results")


def parse_sets(items):
    out = {}
    for it in items or []:
        for kv in it.split():
            k, v = kv.split("=", 1)
            out[k] = yaml.safe_load(v)
            if isinstance(out[k], str):       # "1e9" などを数値に
                try:
                    out[k] = float(out[k])
                except ValueError:
                    pass
    return out


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", help="key=value（空白区切りで複数可）")
    ap.add_argument("--variant", nargs=3, action="append", metavar=("KEY", "NAME", "OVERRIDES"))
    ap.add_argument("--validate", help="仕様12章の検証をかける variant の KEY")
    ap.add_argument("--sims", type=int, default=1000, help="ランダムエントリー比較の試行回数")
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    warnings.filterwarnings("ignore", message=".*スワップが未設定.*")
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base.update(parse_sets(args.set))
    variants = [(k, n, parse_sets([o])) for k, n, o in (args.variant or [("base", "設定ファイルどおり", "")])]

    label = args.label or "_".join(k for k, _, _ in variants)
    out = RESULTS_ROOT / f"{datetime.now():%Y%m%d_%H%M}_{label}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"📁 結果の保存先: {out}")
    t_start = time.time()

    pairs_all = list(base.get("pairs") or available_fx_pairs())
    bars_cache = {}

    def bars_for(hours):
        if hours not in bars_cache:
            bars_cache[hours] = {p: load_bars(p, hours) for p in pairs_all}
        return bars_cache[hours]

    if not (base.get("swap_pips_per_day") or {}):
        print("⚠️ スワップの表（swap_pips_per_day）が空なので、全ペアでスワップを 0 として計算します。")

    results = dict(created=f"{datetime.now():%Y-%m-%d %H:%M}", base=base, variants=[], validation=None)
    summary_rows, pair_rows, year_rows, funnel_rows, loo_rows = [], [], [], [], []
    bts = {}
    for key, name, over in variants:
        cfg = dict(base, **over)
        hours = int(cfg["bar_hours"])
        bars = bars_for(hours)
        print(f"\n▶ {key}: {name}  {over}")
        bt = run_portfolio(cfg, bars)
        bts[key] = (cfg, bt)
        m = all_metrics(bt) if len(bt.equity) else {}
        m.update(key=key, name=name, overrides=over, signals=sum(len(v) for v in bt.signals.values()),
                 skipped=len(bt.skipped))
        summary_rows.append(m)
        print(f"   シグナル {m['signals']} / 約定 {m['trades']} / 見送り {m['skipped']} / 合計R {m['total_r']:+.2f}")
        tr = bt.trades
        tr.to_csv(out / f"trades_{key}.csv", index=False, encoding="utf-8-sig")
        bt.skipped.to_csv(out / f"skipped_{key}.csv", index=False, encoding="utf-8-sig")
        bt.equity.to_csv(out / f"equity_{key}.csv", encoding="utf-8-sig")
        if len(tr):
            tr = tr.assign(year=pd.to_datetime(tr["exit_date"]).dt.year)
        for p in pairs_all:
            sub = tr[tr["pair"] == p] if len(tr) else tr
            pair_rows.append(dict(key=key, pair=p, trades=len(sub), total_r=float(sub["R"].sum()) if len(sub) else 0.0))
        if len(tr):
            for y, g in tr.groupby("year"):
                year_rows.append(dict(key=key, year=int(y), trades=len(g), total_r=float(g["R"].sum())))
        # どこで落ちているか
        start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
        for d in cfg.get("directions", (1, -1)):
            tot = {}
            for p in pairs_all:
                b = bars[p][bars[p].index <= end]
                t_from = int(b.index.searchsorted(start))
                for k, v in diagnostics.funnel(b, cfg, d, t_from).items():
                    tot[k] = tot.get(k, 0) + v
            for stage in diagnostics.ORDER:
                funnel_rows.append(dict(key=key, direction=d, stage=stage, bars=tot.get(stage, 0)))
        loo = diagnostics.leave_one_out({p: bars[p] for p in pairs_all}, cfg, start, end, args.workers)
        for k, v in loo.items():
            loo_rows.append(dict(key=key, relaxed=k, signals=v))
        results["variants"].append(dict(
            key=key, name=name, overrides=over, metrics=m,
            trades=tr.drop(columns=["legs"]).to_dict("records") if len(tr) else [],
            skipped=bt.skipped.to_dict("records"),
            equity=[[str(i.date()) if hours == 24 else str(i), float(r["equity"])] for i, r in bt.equity.iterrows()],
            holding=holding_distribution(tr).to_dict() if len(tr) else {},
            exit_reasons=tr["exit_reason"].value_counts().to_dict() if len(tr) else {},
        ))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pair_rows).to_csv(out / "by_pair.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(year_rows).to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(funnel_rows).to_csv(out / "funnel.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(loo_rows).to_csv(out / "leave_one_out.csv", index=False, encoding="utf-8-sig")
    results.update(by_pair=pair_rows, yearly=year_rows, funnel=funnel_rows, loo=loo_rows)

    # ---- 仕様 12.3〜12.7 の検証 ----
    if args.validate:
        cfg, bt = bts[args.validate]
        bars = bars_for(int(cfg["bar_hours"]))
        print(f"\n▶ 検証（{args.validate}）")
        v = {}
        v["cost"] = validation.cost_sensitivity(cfg, bars, args.workers)
        v["param"] = validation.param_sensitivity(cfg, bars, args.workers)
        wf, wf_trades = validation.walk_forward(cfg, bars, workers=args.workers)
        v["walk_forward"] = wf
        train_pairs = [p for p in ("EURUSD", "GBPUSD", "AUDUSD") if p in pairs_all]
        test_pairs = [p for p in pairs_all if p not in train_pairs]
        v["pair_split"] = pd.DataFrame([validation.pair_split(cfg, bars, train_pairs, test_pairs, args.workers)])
        rnd = [validation.random_benchmark(cfg, bars, bt.trades, args.sims, seed=1, trend_aligned=a) for a in (False, True)]
        for name, df in v.items():
            df.to_csv(out / f"validation_{name}.csv", encoding="utf-8-sig")
        rnd_rows = [{k: x for k, x in r.items() if k != "means"} for r in rnd if r]
        pd.DataFrame(rnd_rows).to_csv(out / "validation_random.csv", index=False, encoding="utf-8-sig")
        cols = ["trades", "total_r", "avg_r", "win_rate", "pf", "max_dd_r"]
        results["validation"] = dict(
            key=args.validate,
            cost=v["cost"][cols].reset_index().to_dict("records"),
            param=v["param"][cols].reset_index().to_dict("records"),
            walk_forward=wf.to_dict("records"),
            walk_forward_oos=dict(trades=len(wf_trades), total_r=float(wf_trades["R"].sum()) if len(wf_trades) else 0.0),
            pair_split=v["pair_split"].to_dict("records"),
            random=[dict({k: x for k, x in r.items() if k != "means"},
                         hist=np.histogram(r["means"][np.isfinite(r["means"])], bins=30)) for r in rnd if r],
        )
        print("   検証 完了")

    # ---- トレード付きのダウ理論チャート ----
    charts = []
    if not args.no_charts:
        from dow_swing_chart import make_chart
        from cwh_fx.src.charts import trades_for_chart
        for key, name, over in variants:
            cfg, bt = bts[key]
            if not len(bt.trades):
                continue
            hours = int(cfg["bar_hours"])
            risk_display = cfg["initial_equity"] * cfg["risk_per_trade"]
            for pair in sorted(bt.trades["pair"].unique()):
                df1, _ = load_1min(pair, "bid")
                sub = bt.trades[bt.trades["pair"] == pair]
                ct = trades_for_chart(sub, df1, hours, bt.costs.spread, risk_display)
                path = out / f"dow_trade_chart_{pair}_{key}.html"
                make_chart(df1, path, f"{pair} {cfg['start_date']} ~ {cfg['end_date']} / カップウィズハンドル: {name}",
                           cfg["start_date"], cfg["end_date"], trades=ct, weekly_trend=None, mode=name,
                           risk=risk_display, entry_window=cfg.get("swing_window", 5), entry_atr=cfg["atr_short"],
                           price_decimals=price_decimals_for(pair), signal_hours=hours,
                           ma_periods=(50, 200))
                charts.append(dict(key=key, pair=pair, file=path.name, trades=len(sub),
                                   mb=round(path.stat().st_size / 1e6, 1)))
                print(f"   チャート: {path.name}")
    results["charts"] = charts

    # ---- 保存 ----
    (out / "results.json").write_text(json.dumps(jsonable(results), ensure_ascii=False, indent=1, default=str),
                                      encoding="utf-8")
    (out / "config.json").write_text(json.dumps(jsonable(dict(base=base, variants=variants, validate=args.validate)),
                                                ensure_ascii=False, indent=1), encoding="utf-8")
    code = out / "code"
    shutil.copytree(Path(__file__).resolve().parent, code / "cwh_fx",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    from cwh_fx.report_html import build_report
    build_report(out)
    print(f"\n✅ 完了（{time.time() - t_start:.0f}秒）: {out / 'report.html'}")


if __name__ == "__main__":
    main()
