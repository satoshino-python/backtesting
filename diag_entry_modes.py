"""
エントリー方式（4H高安 / 4Hスイング / 1Hスイング）の違いを測る診断。ER(20)≥0.3 かつ MA整列で、その方向が許可されている1時間足バーについて、
  - 注文が置ける割合（ラインが有効で、始値がラインより内側にある）
  - 注文のラインと始値の距離（4時間足ATR単位）の中央値
を方式ごとに集計する。1時間足バー単位の近似（エンジンは1分足単位）。結果は CSV に保存。
使い方: python diag_entry_modes.py
"""
import contextlib, io, warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import numpy as np, pandas as pd
import main_4H_fixedSL_multi as multi
import main_4H_trend_breakout_fast as m
from main_4H_fixedSL import (load_gmo_click_1min_data, resample_to_signal_bars, compute_signals,
                             get_trading_day_label, get_signal_bar_label)
from main_4H_fixedSL_multi import zip_month_range, price_decimals_for


def one(pair):
    multi.WARMUP_MONTHS = m.WARMUP_MONTHS
    cfg = replace(m.BASE_CFG, data_path=m.DATA_ROOT / pair, price_decimals=price_decimals_for(pair))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s, e = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
        df, _ = load_gmo_click_1min_data(cfg.data_path, price_side=cfg.price_side, price_decimals=cfg.price_decimals,
                                         month_range=zip_month_range(cfg), spread_period=(s, e), session_start_hour=cfg.session_start_hour)
        h1 = resample_to_signal_bars(df, 17, 1); h4 = resample_to_signal_bars(df, 17, 4); d1 = resample_to_signal_bars(df, 17, 24)
        atr4 = m.atr_wilder(h4["High"], h4["Low"], h4["Close"], 14, cfg.price_decimals).shift(1).reindex(get_signal_bar_label(h1.index, 17, 4)).to_numpy()
        c = d1["Close"]; ma_s, ma_m, ma_l = (c.rolling(p).mean() for p in m.MA_PERIODS); er = m.efficiency_ratio(c, 20)
        f = pd.DataFrame({"up": (ma_s > ma_m) & (ma_m > ma_l), "dn": (ma_s < ma_m) & (ma_m < ma_l), "ok": ma_l.notna() & (er >= 0.3)}).shift(1)
        f = f.reindex(get_signal_bar_label(h1.index, 17, 24))
        ok = f["ok"].fillna(False).to_numpy(bool); up = f["up"].fillna(False).to_numpy(bool) & ok; dn = f["dn"].fillna(False).to_numpy(bool) & ok
        day = get_trading_day_label(h1.index, 17); inp = np.asarray((day >= s) & (day <= e))
        sigs = {}
        d = m.breakout_signals(h4, 18, cfg.price_decimals).reindex(get_signal_bar_label(h1.index, 17, 4)); sigs["4H 高安18"] = d
        sg = compute_signals(h4, window=18, atr_period=14, price_decimals=cfg.price_decimals).reindex(get_signal_bar_label(h1.index, 17, 4)); sigs["4H スイング18"] = sg
        sigs["1H スイング24"] = compute_signals(h1, window=24, atr_period=14, price_decimals=cfg.price_decimals)
        rows = []
        for name, sg in sigs.items():
            sh, sl = sg["SH"].to_numpy(float), sg["SL"].to_numpy(float)
            shv, slv = sg["SHValid"].fillna(1).to_numpy(float) != 0, sg["SLValid"].fillna(1).to_numpy(float) != 0
            o = h1["Open"].to_numpy(float)
            can_l = up & inp & shv & (o < sh); can_s = dn & inp & slv & (o > sl)
            dist = np.r_[((sh - o) / atr4)[can_l], ((o - sl) / atr4)[can_s]]
            rows.append(dict(Pair=pair, Mode=name, AllowedBars=int((up | dn)[inp].sum()), CanPlace=int(can_l.sum() + can_s.sum()),
                             ValidShare=float(np.mean(np.r_[shv[up & inp], slv[dn & inp]])), DistMed=float(np.nanmedian(dist)) if len(dist) else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    with ProcessPoolExecutor(m.WORKERS) as ex:
        t = pd.concat(list(ex.map(one, m.PAIRS)), ignore_index=True)
    t.to_csv("trend_results/diag_entry_modes.csv", index=False)
    g = t.groupby("Mode").apply(lambda x: pd.Series(dict(
        可注文の割合=x.CanPlace.sum() / x.AllowedBars.sum() * 100, ラインが有効な割合=(x.ValidShare * x.AllowedBars).sum() / x.AllowedBars.sum() * 100,
        距離中央値_ATR=x.DistMed.median())))
    print(g.round(2))
