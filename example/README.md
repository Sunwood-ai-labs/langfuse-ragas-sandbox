# LangfuseとRagasによるRAG評価実験

このディレクトリには、LangfuseとRagasを使用してRAGシステムを評価するための一連のサンプル実装が含まれています。架空のマーダーミステリーデータセットを使用して、段階的に評価手法を学ぶことができます。

## 📁 ディレクトリ構成

```
.
├── 01_basic_setup/           # 基本的なRAGシステムの実装
│   ├── requirements.txt      # 必要なパッケージリスト
│   ├── .env.example         # 環境変数設定例
│   ├── basic_rag.py         # RAGシステムの基本実装
│   └── README.md            # 基本実装の説明
│
├── 02_ragas_evaluation/      # Ragasによる標準評価
│   ├── evaluate_rag.py      # Ragas評価の実装
│   └── README.md            # Ragas評価の説明
│
├── 03_advanced_evaluation/   # 高度な評価手法
│   ├── custom_metrics.py    # カスタム評価指標の実装
│   └── README.md            # カスタム評価の説明
│
└── data/                    # データセット
    └── mystery_dataset.json # マーダーミステリーデータ
```

## 🎯 学習ステップ

### Step 1: 基本的なRAGシステムの構築
- `01_basic_setup/`ディレクトリで、基本的なRAGシステムを実装
- GPT-4とGPT-4 Turbo Previewの両方のモデルで評価
- Langfuseによる基本的なトレース記録

### Step 2: Ragasによる標準評価
- `02_ragas_evaluation/`ディレクトリで、標準的な評価指標を実装
- Faithfulness、AnswerRelevancy、ContextRelevancyなどの評価
- 評価結果のLangfuseでの可視化

### Step 3: カスタム評価指標の実装
- `03_advanced_evaluation/`ディレクトリで、独自の評価指標を実装
- ミステリー特有の要素（動機、アリバイなど）の評価
- 説明の一貫性や論理性の評価

## 🚀 使用方法

1. 環境のセットアップ
```bash
cd 01_basic_setup
cp .env.example .env
uv pip install -r requirements.txt
```

2. 基本的なRAG評価の実行
```bash
cd 01_basic_setup
python basic_rag.py
```

3. Ragas評価の実行
```bash
cd ../02_ragas_evaluation
python evaluate_rag.py
```

4. カスタム評価の実行
```bash
cd ../03_advanced_evaluation
python custom_metrics.py
```

## 📊 評価指標の概要

### 標準的なRagas評価指標
- Faithfulness（忠実性）
- AnswerRelevancy（回答の関連性）
- ContextRelevancy（文脈の関連性）
- AspectCritique（多面的評価）

### カスタム評価指標
- MysteryRelevanceMetric（ミステリー要素の関連性）
- NarrativeCoherenceMetric（説明の一貫性）

## 💡 Langfuseでの確認方法

1. Langfuseダッシュボードにアクセス
2. トレース一覧で各実行結果を確認
3. メトリクスタブで評価スコアを確認
4. モデル間での比較分析が可能

## ⚠️ 注意事項

- APIキーは必ず`.env`ファイルで管理
- 大量のAPIコールを行う場合はコストに注意
- 評価結果は相対的な比較として使用することを推奨
- 定期的な評価の実行で、パフォーマンスの変化を監視

## 📚 参考リンク

- [Langfuse Documentation](https://langfuse.com/docs)
- [Ragas Documentation](https://docs.ragas.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
