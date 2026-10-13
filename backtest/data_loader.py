"""1分足(GMO) → 日足(NY17:00区切り・取引日ラベル)。"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from main_4H_fixedSL import load_gmo_click_1min_data, get_trading_day_label


def load_1min(pair):
    df, spread = load_gmo_click_1min_data(f"histData/{pair}", price_side="mid")
    return df, spread


def to_daily(df_1min):
    """NY17:00〜翌17:00 を1日とした日足。インデックス=取引日(月〜金)。終値の時刻はその日の NY 17:00。"""
    lab = get_trading_day_label(df_1min.index, 17)
    d = df_1min.groupby(lab).agg(Open=("Open", "first"), High=("High", "max"),
                                 Low=("Low", "min"), Close=("Close", "last"))
    return d[d.index.dayofweek < 5]
