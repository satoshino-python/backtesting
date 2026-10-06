# CLAUDE.md

FX（GMOクリック証券の1分足）のスイングブレイクアウト戦略を backtesting.py で検証するリポジトリ。
各スクリプトのルール・設定・出力は [docs/STRATEGIES.md](docs/STRATEGIES.md)、ダウ理論判定の仕様は
[docs/DOW_TREND_SPEC.md](docs/DOW_TREND_SPEC.md) にまとめてある。作業の前に読むこと。

## 実行
- リポジトリのルートで実行する（`histData/` などを相対パスで探す）。ローカルの Python は `.venv/`。
- Windows では出力の絵文字で落ちるため `PYTHONIOENCODING=utf-8` を付ける。
- 1分足で5年分は時間がかかる（1回あたり数分）。新しいルールは短い期間で確かめてから広げる。
- `main_4H_fixedSL.py` は他のスクリプトから import されている。変更すると multi / dow 版の結果も変わる。
- `fast_engine.py` は `SwingBreakoutStrategy1Min`（固定SL/TP）と `SwingBreakoutTrail1Min`（1時間足トレーリング）の売買ルールを
  backtesting.py なしで再現した高速版（約100倍速い）。複数ペアは `main_4H_fixedSL_fast_multi.py` で並列に検証できる。
  戦略のルールを変えたら `fast_engine.py` も合わせて直し、`compare_fast_engine.py` で backtesting.py との一致を確かめる。

## 検証結果のチャート（毎回作る）
トレードを伴う検証では、backtesting.py 標準のチャートに加えて、**トレード付きのダウ理論チャート**を必ず出力する。
`dow_swing_chart.make_chart()` を呼ぶだけでよい（`main_4H_fixedSL_dow.py` は `TRADE_CHART = True` で自動的に出力する）。

```python
from dow_swing_chart import make_chart

make_chart(
    df_1min,                      # load_gmo_click_1min_data() の1分足。助走期間を含む全期間
    out_dir / f"dow_trade_chart_{symbol}_{period}_{mode}.html",
    f"{symbol} {start} ~ {end} / トレード: {mode}",
    start, end,
    trades=stats["_trades"],      # backtesting.py の取引履歴（そのCSVのパスでも可）
    weekly_trend=used_weekly,     # 各週に使った週足トレンド（前週末の判定）。使わない戦略は None
    mode=mode, risk=cfg.cash * cfg.risk_pct,
    entry_window=cfg.window, entry_atr=cfg.atr_period, price_decimals=cfg.price_decimals,
)
```

- 表示内容: エントリー（▲買い / ▼売り）・決済（●）・SL/TP、背景の週足トレンド、4時間足のエントリーライン、前/次のトレード移動。
- 既存の取引履歴CSVからは `python dow_swing_chart.py --trades dow_results/trades_<期間>_<モード>.csv` で作れる。
- チャートのテンプレートは `dow_swing_chart_template.html`。JavaScript の `computeDow` は `dow_trend.py` の移植なので、
  片方を変えたらもう片方も合わせる。

## 結果の保存
- 実行ごとに結果を別フォルダ・別ファイル名に保存し、前の結果を上書きしない（`multi_results/<日時>_<ラベル>/` など）。
- 設定を一時的に変えて実行するときは、スクリプトの `CFG` を書き換えずに `dataclasses.replace` で上書きして呼び出す。
