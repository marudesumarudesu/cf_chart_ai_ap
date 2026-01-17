# JP Market Canvas (Streamlit)

yfinance をデータソースにした、日本株チャートダッシュボードです。

- 複数銘柄をローソク足で同時表示（最大90本）
- 重要指数の表示（例：日経平均、TOPIX、USD/JPY、S&P500 等）
- 選択銘柄から「平均インデックス（Equal-Weight）」を作成して表示
- 10種類以上のテクニカル指標（SMA/EMA/BB/一目均衡表/RSI/MACD ...）
- JPXが公開している銘柄一覧Excelを読み込み、検索して全銘柄から選択可能

> **注意**: 本アプリは情報提供が目的です。投資助言・売買推奨ではありません。

---

## ローカル実行

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

---

## GitHub → Streamlit Cloud でデプロイ

1. このリポジトリを GitHub に push
2. Streamlit Community Cloud にログイン
3. "New app" → GitHub リポジトリを選択
4. **Main file path** を `app.py` に指定して Deploy

---

## 使い方のコツ（yfinanceの制限対策）

- **銘柄数は最初は 3〜8 銘柄**がおすすめです（増やすほど重くなります）
- 本アプリは Streamlit の `st.cache_data` で **15分キャッシュ**します
- 取得期間を必要以上に長くしない（指標用に 420日程度をデフォルトにしています）

---

## 指標一覧

- 価格に重ねる：SMA / EMA / Bollinger Bands / Ichimoku / VWAP / Parabolic SAR / Supertrend
- 別パネル：RSI / MACD / Stochastic / ATR / ADX / OBV / CCI / Williams %R

---

## ライセンス

MIT（お好みで変更してください）
