# legacy

backtesting.py でしか動かない古いスクリプト。今の検証は `fast_engine.py`（ルートのスクリプト）で行う。

| ファイル | 内容 |
|---|---|
| `main_D1.py` | 最初の版。日足のスイングブレイク、固定枚数 |
| `main_4H.py` | 4時間足（任意の時間足）版、固定枚数。SL×TP グリッド検証 |

- 実行はリポジトリのルートから（`histData/` を相対パスで探す）: `python legacy/main_4H.py`
- `pip install backtesting==0.6.6`（チャートの取引マークを出すには `bokeh==2.4.3`）が必要
- ルールの詳細は [docs/STRATEGIES.md](../docs/STRATEGIES.md) の 4.1・4.2
