"""テスト用の合成データ（日足）"""
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


def load_params(**over):
    p = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p.update(over)
    return p


def bars_from_close(close, ranges):
    """終値の列と値幅から OHLC を作る（始値 = 前日終値、高値・安値は始値と終値を包む）"""
    close = np.asarray(close, dtype=float)
    ranges = np.broadcast_to(np.asarray(ranges, dtype=float), close.shape)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + ranges / 2
    low = np.minimum(open_, close) - ranges / 2
    idx = pd.bdate_range("2020-01-01", periods=len(close))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def make_cup(kind="u", depth=0.035, cup_len=60, handle_len=10, handle_pull=0.006, breakout=True,
             handle_range=0.0012, range_=0.005, tail=15):
    """
    上昇トレンド → カップ（kind="u" は丸底、"v" は V字）→ ハンドル → ブレイク → その後の上昇。
    返り値: (df, 情報 dict: i_left, handle_end, breakout)
    """
    up = np.linspace(1.00, 1.20, 260)                    # SMA200 が上向きになる長い上昇（左縁の手前で終わる）
    k = np.arange(1, cup_len + 1) / cup_len
    if kind == "u":
        cup = 1.20 - depth * np.sin(np.pi * k)
    else:
        cup = 1.20 - depth * (1 - np.abs(2 * k - 1))
    rim = cup[-1]
    # ハンドル: 最初の4本で handle_pull だけ押し、その後は小さく戻しながら横ばい
    hk = np.arange(1, handle_len + 1)
    handle = rim - handle_pull * np.minimum(hk / 4, 1) + np.maximum(hk - 4, 0) * 0.0001
    close = np.r_[up, cup, handle]
    # 値動きはカップの右側にかけて小さくなる（ボラティリティの収縮）
    cup_ranges = np.linspace(range_, handle_range, cup_len)
    ranges = np.r_[np.full(len(up), range_), cup_ranges, np.full(handle_len, handle_range)]
    i_left = len(up) - 1
    handle_end = len(close) - 1
    info = dict(i_left=i_left, handle_end=handle_end, breakout=None)
    if breakout:
        post = rim + 0.004 + np.arange(tail) * 0.002
        close = np.r_[close, post]
        ranges = np.r_[ranges, np.full(tail, range_)]
        info["breakout"] = handle_end + 1
    return bars_from_close(close, ranges), info
