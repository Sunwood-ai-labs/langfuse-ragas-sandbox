
<div>

![Image](https://github.com/user-attachments/assets/61642ed5-6ac1-4753-848e-9935ab99d2a0)

# langfuse-ragas-sandbox

</div>

LangfuseとRagasを使用して、LLMアプリケーションの評価を行うためのサンプル実装を提供するリポジトリです。架空のマーダーミステリーデータセットを使用して、段階的に評価手法を学ぶことができます。

## 🌟 特徴

- GPT-4とGPT-4 Turbo Previewの両方のモデルでの評価
- Ragasによる標準的な評価指標の実装
- ミステリー特化のカスタム評価指標
- Langfuseによる評価結果の可視化
- 段階的な学習が可能な構造化された実装例

## 📁 リポジトリ構成

```
.
├── example/                  # 評価実験のメインディレクトリ
│   ├── 01_basic_setup/      # 基本的なRAGシステム
│   ├── 02_ragas_evaluation/ # 標準的な評価手法
│   ├── 03_advanced_evaluation/ # カスタム評価指標
│   └── data/               # テストデータセット
│
├── README.md               # このファイル
└── .gitignore
```

詳細な実装とドキュメントは[example/README.md](example/README.md)を参照してください。

## 🎯 機能概要

### 1. 基本的なRAGシステム（01_basic_setup）
- マーダーミステリーデータセットを使用したRAG実装
- GPT-4とGPT-4 Turbo Previewでの比較評価
- Langfuseによる基本的なトレース記録

### 2. Ragas評価（02_ragas_evaluation）
- Faithfulness（忠実性）
- AnswerRelevancy（回答の関連性）
- ContextRelevancy（文脈の関連性）
- AspectCritique（多面的評価）

### 3. カスタム評価（03_advanced_evaluation）
- MysteryRelevanceMetric（ミステリー要素の関連性）
  - 動機、アリバイ、証拠などの評価
- NarrativeCoherenceMetric（説明の一貫性）
  - 論理的な説明の流れを評価

## 🚀 クイックスタート

```bash
# リポジトリのクローン
git clone https://github.com/yourusername/langfuse-ragas-sandbox.git
cd langfuse-ragas-sandbox

# 環境設定
cd example/01_basic_setup
cp .env.example .env
uv pip install -r requirements.txt

# 基本評価の実行
python basic_rag.py
```

詳細な実行手順は各ディレクトリのREADMEを参照してください。

## 📊 評価結果の確認

1. Langfuseダッシュボード（http://localhost:3000）にアクセス
2. 以下の情報を確認可能：
   - 各モデルの回答品質
   - 評価指標ごとのスコア
   - トレース機能による詳細分析

## ⚠️ 前提条件

- Python 3.10以上
- uvパッケージマネージャー
- OpenAI API キー
- Langfuse（ローカルまたはクラウド）

## 📝 環境変数の設定

```env
LANGFUSE_PUBLIC_KEY=your-public-key
LANGFUSE_SECRET_KEY=your-secret-key
LANGFUSE_HOST=http://localhost:3000
OPENAI_API_KEY=your-openai-api-key
```

## 👥 コントリビューション

1. このリポジトリをフォーク
2. 新しいブランチを作成
3. 変更をコミット
4. プルリクエストを作成

## 📚 参考リンク

- [Langfuse Documentation](https://langfuse.com/docs)
- [Ragas Documentation](https://docs.ragas.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)

## 📄 ライセンス

[MIT License](LICENSE)
