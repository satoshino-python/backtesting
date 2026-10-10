"""
検証（仕様 12.3〜12.7）:
  3. コスト感度: スプレッド2倍
  4. パラメータ感度: カップの深さの範囲・trail_atr_mult・ハンドル長を ±20%
  5. ウォークフォワード: 学習3年 → 検証1年を繰り返す。パラメータは学習期間の合計Rで選ぶ
  6. ランダムエントリーとの比較: 同じ回数・同じ決済ルール
  7. 銘柄分割: 一部のペアでパラメータを選び、残りのペアで確かめる
"""
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .backtest import run_portfolio, simulate_trade
from .costs import CostModel
from .indicators import atr_wilder
from .pattern import prepare, trend_ok
from .report import all_metrics

SCALES = (0.8, 1.0, 1.2)


def scaled(cfg, depth=1.0, trail=1.0, handle=1.0):
    """主要パラメータを倍率で動かした設定"""
    c = dict(cfg)
    c["cup_depth_min_atr"] = cfg["cup_depth_min_atr"] * depth
    c["cup_depth_max_atr"] = cfg["cup_depth_max_atr"] * depth
    c["trail_atr_mult"] = cfg["trail_atr_mult"] * trail
    c["handle_len_min"] = max(1, int(round(cfg["handle_len_min"] * handle)))
    c["handle_len_max"] = int(round(cfg["handle_len_max"] * handle))
    return c


def _run(args):
    name, cfg, bars = args
    bt = run_portfolio(cfg, bars, log=lambda *a: None)
    m = all_metrics(bt)
    m["name"] = name
    return m, bt.trades


def run_many(items, bars, workers=4):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run, [(n, c, bars) for n, c in items]))


def cost_sensitivity(cfg, bars, workers=4):
    items = [("基準", cfg), ("スプレッド2倍", dict(cfg, spread_mult=2.0)),
             ("スプレッド2倍＋スリッページ2倍", dict(cfg, spread_mult=2.0, slippage_mult=2.0))]
    return pd.DataFrame([m for m, _ in run_many(items, bars, workers)]).set_index("name")


def param_sensitivity(cfg, bars, workers=4):
    items = [("基準", cfg)]
    for k, label in (("depth", "深さの範囲"), ("trail", "trail_atr_mult"), ("handle", "ハンドル長")):
        for s in (0.8, 1.2):
            items.append((f"{label} ×{s}", scaled(cfg, **{k: s})))
    return pd.DataFrame([m for m, _ in run_many(items, bars, workers)]).set_index("name")


def grid(cfg):
    out = []
    for d, t, h in itertools.product(SCALES, SCALES, SCALES):
        out.append((f"depth×{d} trail×{t} handle×{h}", scaled(cfg, d, t, h)))
    return out


def _pick(results):
    """学習期間の合計Rが最大の組み合わせ（同点なら基準に近い方 = 名前に ×1.0 が多い方）"""
    best = max(results, key=lambda m: (m["total_r"], m["name"].count("×1.0")))
    return best["name"]


def walk_forward(cfg, bars, train_years=3, test_years=1, workers=4):
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    rows, oos_trades = [], []
    g = grid(cfg)
    y = start.year
    while True:
        tr0, tr1 = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + train_years - 1}-12-31")
        te0, te1 = pd.Timestamp(f"{y + train_years}-01-01"), pd.Timestamp(f"{y + train_years + test_years - 1}-12-31")
        if te0 > end:
            break
        te1 = min(te1, end)
        train = [m for m, _ in run_many([(n, dict(c, start_date=str(tr0.date()), end_date=str(tr1.date()))) for n, c in g],
                                        bars, workers)]
        pick = _pick(train)
        best_cfg = dict(dict(g)[pick], start_date=str(te0.date()), end_date=str(te1.date()))
        base_cfg = dict(cfg, start_date=str(te0.date()), end_date=str(te1.date()))
        (mt, tt), (mb, _) = run_many([("選んだ設定", best_cfg), ("基準", base_cfg)], bars, workers)
        tr_best = next(m for m in train if m["name"] == pick)
        rows.append(dict(train=f"{tr0.year}〜{tr1.year}", test=f"{te0.year}" if test_years == 1 else f"{te0.year}〜{te1.year}",
                         pick=pick, train_trades=tr_best["trades"], train_r=tr_best["total_r"],
                         test_trades=mt["trades"], test_r=mt["total_r"], base_test_trades=mb["trades"],
                         base_test_r=mb["total_r"]))
        if len(tt):
            oos_trades.append(tt)
        y += test_years
    return pd.DataFrame(rows), (pd.concat(oos_trades) if oos_trades else pd.DataFrame())


def pair_split(cfg, bars, train_pairs, test_pairs, workers=4):
    g = grid(cfg)
    train = [m for m, _ in run_many([(n, dict(c, pairs=list(train_pairs))) for n, c in g], bars, workers)]
    pick = _pick(train)
    (mt, _), (mb, _) = run_many([("選んだ設定", dict(dict(g)[pick], pairs=list(test_pairs))),
                                 ("基準", dict(cfg, pairs=list(test_pairs)))], bars, workers)
    tb = next(m for m in train if m["name"] == pick)
    return dict(train_pairs=", ".join(train_pairs), test_pairs=", ".join(test_pairs), pick=pick,
                train_trades=tb["trades"], train_r=tb["total_r"], test_trades=mt["trades"], test_r=mt["total_r"],
                base_test_trades=mb["trades"], base_test_r=mb["total_r"])


def random_benchmark(cfg, bars, trades, n_sims=1000, seed=0, trend_aligned=False):
    """
    実際と同じ回数のランダムエントリーを、同じ決済ルールで n_sims 回繰り返し、平均Rの分布と比べる。
      - エントリー: 検証期間内のランダムな足の始値。ペアはランダム。買い/売りの比率は実際の取引と同じ確率で選ぶ
      - 損切り幅: 実際の取引の「損切り幅 ÷ ATR14」から無作為に選ぶ
      - trend_aligned=True なら、前の足で SMA200 のトレンド条件（仕様 4.1）を満たす向きの足だけから選ぶ
    """
    if trades is None or len(trades) == 0:
        return None
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(cfg["start_date"]), pd.Timestamp(cfg["end_date"])
    pairs = [p for p in (cfg.get("pairs") or list(bars)) if p in bars]
    costs = CostModel(cfg, pairs)
    p_long = float((trades["direction"] == 1).mean())
    stop_mults = trades["stop_atr"].to_numpy()
    universe = []                     # (pair, i, direction or 0)
    data = {}
    for p in pairs:
        b = bars[p][bars[p].index <= end]
        a14 = atr_wilder(b["high"], b["low"], b["close"], cfg["atr_short"])
        data[p] = (b, a14)
        ok = np.flatnonzero((b.index >= start) & np.isfinite(np.r_[np.nan, a14[:-1]]))
        ok = ok[(ok >= 1) & (ok < len(b) - 1)]
        if trend_aligned:
            for d in (1, -1):
                ind = prepare(b, cfg, d)
                universe += [(p, int(i), d) for i in ok if trend_ok(ind, cfg, int(i) - 1)]
        else:
            universe += [(p, int(i), 0) for i in ok]
    if trend_aligned:
        by_dir = {d: [u for u in universe if u[2] == d] for d in (1, -1)}
    n = len(trades)
    cache = {}
    means = np.empty(n_sims)
    for k in range(n_sims):
        rs = []
        for _ in range(n):
            d = 1 if rng.random() < p_long else -1
            pool = by_dir[d] if trend_aligned else universe
            if not pool:
                continue
            pair, i, _ = pool[rng.integers(len(pool))]
            sm = float(stop_mults[rng.integers(len(stop_mults))])
            key = (pair, i, d, round(sm, 3))
            if key not in cache:
                b, a14 = data[pair]
                cache[key] = simulate_trade(b, pair, i, d, sm, cfg, costs, a14)[0]
            rs.append(cache[key])
        means[k] = np.mean(rs) if rs else np.nan
    actual = float(trades["R"].mean())
    return dict(actual_avg_r=actual, n_trades=n, n_sims=n_sims, trend_aligned=trend_aligned,
                rand_mean=float(np.nanmean(means)), rand_p5=float(np.nanpercentile(means, 5)),
                rand_p50=float(np.nanpercentile(means, 50)), rand_p95=float(np.nanpercentile(means, 95)),
                p_value=float(np.mean(means >= actual)), means=means)
