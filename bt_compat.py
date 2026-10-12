"""
backtesting.py を「入っていれば使う」ための import。

通常の検証は fast_engine で行うため、backtesting.py（と Bokeh）が無くても動くようにしてある。
backtesting.py 版の Strategy クラスはこのモジュールの Strategy を継承して定義するので、
入っていなければ定義だけされ、Backtest を作ろうとした時点でエラーになる。

答え合わせ（compare_fast_engine.py）や backtesting.py 標準のチャートには backtesting.py 0.6.6 が必要:
    pip install backtesting==0.6.6
"""
try:
    from backtesting import Backtest, Strategy
    HAVE_BACKTESTING = True
except ImportError:
    HAVE_BACKTESTING = False

    class Strategy:
        """backtesting.py が無いときの置き換え（クラス定義のためだけに使う）"""

    class Backtest:
        def __init__(self, *args, **kwargs):
            raise ImportError("backtesting.py が入っていません。fast_engine を使うか、"
                              "pip install backtesting==0.6.6 で入れてください")


_warned = False


def warn_bokeh_version():
    """backtesting.py のチャートを作る前に呼ぶ。Bokeh 3.x では取引マーク（▲/▼）が表示されない"""
    global _warned
    if _warned:
        return
    _warned = True
    try:
        import bokeh
    except ImportError:
        return
    if bokeh.__version__.startswith("3."):
        print("=" * 60)
        print(f"⚠️  検出された Bokeh バージョン: {bokeh.__version__}")
        print("⚠️  backtesting.py は Bokeh 3.x で取引マーク（▲/▼矢印）が非表示になる問題があります。")
        print("👉 ターミナルで以下のコマンドを実行して Bokeh 2.4.3 をインストールしてください:")
        print("   pip install \"bokeh==2.4.3\"")
        print("=" * 60 + "\n")
