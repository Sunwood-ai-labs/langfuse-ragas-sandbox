# カスタム評価指標による高度な評価

このディレクトリでは、マーダーミステリー特有の要素を評価するためのカスタム評価指標を実装しています。

## 🎯 カスタム評価指標

### 1. MysteryRelevanceMetric（ミステリー要素の関連性）

以下の要素への言及度を評価します：
- 動機
- アリバイ
- 証拠
- 時系列
- 容疑者
- 犯行手法
- トリック

各要素について0（言及なし）から1（十分な言及あり）のスコアを算出し、総合評価を行います。

### 2. NarrativeCoherenceMetric（説明の一貫性）

以下の観点から回答の論理的一貫性を評価します：
- 説明の順序は論理的か
- 因果関係は明確か
- 矛盾する記述はないか
- 結論に至るまでの推論過程は妥当か

## 🚀 使用方法

```bash
python custom_metrics.py
```

## 💡 実装の特徴

1. LLMを活用した評価
   - 各要素の評価にGPT-4を使用
   - テキスト埋め込みによる類似度分析

2. 詳細な評価レポート
   - 要素ごとの個別スコア
   - 総合評価スコア
   - 評価理由の説明

3. Langfuseとの連携
   - カスタム評価指標もLangfuseで追跡可能
   - 時系列での変化を監視
   - モデル間の比較分析

## 📊 評価結果の解釈

### MysteryRelevanceMetricのスコア
- 0.0-0.3: 不十分な言及
- 0.3-0.7: 部分的な言及
- 0.7-1.0: 十分な言及

### NarrativeCoherenceMetricのスコア
- 0.0-0.3: 論理的一貫性が低い
- 0.3-0.7: 部分的に一貫している
- 0.7-1.0: 高い論理的一貫性

## 🔧 カスタマイズ方法

1. 新しい評価要素の追加
```python
self.mystery_elements.append("新しい要素")
```

2. 評価基準の調整
```python
# スコアの重み付けを変更
weights = {
    "動機": 0.3,
    "証拠": 0.3,
    "アリバイ": 0.2,
    # ...
}
```

3. 評価プロンプトの修正
- `custom_metrics.py`内の各評価プロンプトを必要に応じて調整可能

## ⚠️ 注意事項

- カスタム評価には追加のAPIコールが必要となるため、コストに注意
- 評価基準は定期的に見直しと調整を推奨
- 結果の解釈には、コンテキストや用途に応じた判断が必要
