#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import sys
from typing import List, Dict
from dotenv import load_dotenv
import asyncio
import uuid
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from loguru import logger
from tqdm import tqdm

# loguruの設定
logger.remove()  # デフォルトのハンドラを削除
logger.add(
    "logs/evaluation_{time}.log",
    rotation="1 day",
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(sys.stderr, level="INFO")  # 標準エラー出力にも表示

def check_required_env_vars():
    """必要な環境変数が設定されているか確認"""
    required_vars = [
        "OPENAI_API_KEY",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST"
    ]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"以下の環境変数が設定されていません: {', '.join(missing_vars)}")

def load_mystery_dataset() -> Dict:
    """ミステリーデータセットを読み込む"""
    logger.info("ミステリーデータセットの読み込みを開始")
    dataset_path = os.path.join(os.path.dirname(__file__), "../data/mystery_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"データセットファイルが見つかりません: {dataset_path}")
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("データセットの形式が正しくありません。'cases'キーが必要です。")
        
    logger.success(f"データセットの読み込みが完了: {len(data['cases'])}件のケースを読み込みました")
    return data

def prepare_evaluation_dataset(dataset: Dict) -> Dataset:
    """評価用のデータセットを準備"""
    logger.info("評価用データセットの準備を開始")
    if not dataset.get("cases"):
        raise ValueError("データセットにケースが含まれていません")

    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truths": []
    }
    
    for case in tqdm(dataset["cases"], desc="データセット準備"):
        if not case.get("clue_questions"):
            logger.warning(f"ケース {case.get('id', 'unknown')} に質問が含まれていません")
            continue
            
        for q in case["clue_questions"]:
            eval_data["question"].append(q)
            
            # コンテキストの準備（証拠と容疑者情報）
            contexts = case.get("evidence", []).copy()
            for suspect in case.get("suspects", []):
                contexts.append(
                    f"{suspect['name']}({suspect['role']})のアリバイ: {suspect['alibi']}"
                )
            eval_data["contexts"].append(contexts)
            
            # ground truthとしてcase["solution"]を使用
            if not case.get("solution"):
                logger.warning(f"ケース {case.get('id', 'unknown')} に解決策が含まれていません")
            eval_data["ground_truths"].append([case.get("solution", "")])
            
            # 一時的な回答として空文字を設定
            eval_data["answer"].append("")
    
    if not eval_data["question"]:
        raise ValueError("有効な評価データが作成できませんでした")
        
    dataset = Dataset.from_dict(eval_data)
    logger.success(f"データセットの準備が完了: {len(dataset)}件の評価データを作成")
    return dataset

def init_ragas_metrics(metrics, llm, embedding):
    """Ragasメトリクスの初期化"""
    logger.info("Ragasメトリクスの初期化を開始")
    for metric in tqdm(metrics, desc="メトリクス初期化"):
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embedding
        run_config = RunConfig()
        metric.init(run_config)
    logger.success("メトリクスの初期化が完了")

def setup_ragas_evaluator(llm, embeddings):
    """Ragas評価器のセットアップ"""
    logger.info("Ragas評価器のセットアップを開始")
    # 評価メトリクスの設定
    metrics = [
        Faithfulness(),  # 忠実性：背景情報と一貫性のある回答ができているか
        ResponseRelevancy(),  # 関連性：質問と関連した回答ができているか
        LLMContextRecall(),  # 文脈精度：質問や正解に関連した背景情報を取得できているか
        LLMContextPrecisionWithoutReference(),  # 文脈回収：回答に必要な背景情報をすべて取得できているか
    ]
    
    # メトリクスの初期化
    init_ragas_metrics(
        metrics,
        llm=LangchainLLMWrapper(llm),
        embedding=LangchainEmbeddingsWrapper(embeddings)
    )
    
    logger.success("評価器のセットアップが完了")
    return metrics

async def evaluate_with_model(model_name: str, eval_dataset: Dataset, metrics, langfuse, handler):
    """指定されたモデルで評価を実行"""
    logger.info(f"{model_name}による評価を開始")
    llm = ChatOpenAI(model=model_name, temperature=0)
    embeddings = OpenAIEmbeddings()

    # RAGチェーンの設定
    prompt = ChatPromptTemplate.from_template(
        "背景情報をもとに質問に回答してください。背景情報： {context} 質問： {question}"
    )

    # メトリクスの設定
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = LangchainLLMWrapper(llm)
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = LangchainEmbeddingsWrapper(embeddings)
        run_config = RunConfig()
        metric.init(run_config)

    results = {}
    for i, sample in enumerate(tqdm(eval_dataset, desc=f"{model_name}の評価進捗")):
        # トレースIDの生成
        trace_id = str(uuid.uuid4())

        # 実際の質問応答を実行
        vectorstore = FAISS.from_texts(texts=sample["contexts"], embedding=embeddings)
        retriever = vectorstore.as_retriever()
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 回答の生成
        answer = chain.invoke(
            sample["question"],
            config={
                "run_id": trace_id,
                "callbacks": [handler],
                "metadata": {"ground_truth": sample["ground_truths"][0]},
            }
        )

        logger.info(f"質問: {sample['question']}")
        logger.info(f"回答: {answer}")

        # 各メトリクスで評価
        sample_scores = {}
        for metric in metrics:
            ragas_sample = SingleTurnSample(
                question=sample["question"],
                contexts=sample["contexts"],
                retrieved_contexts=sample["contexts"],
                answer=answer,
                response=answer,
                user_input=sample["question"],
                reference=sample["ground_truths"][0]
            )
            score = await metric.single_turn_ascore(ragas_sample)
            sample_scores[metric.name] = score
            logger.info(f"{metric.name}のスコア: {score}")

            # Langfuseにスコアを記録
            langfuse.score(
                name=f"{model_name}_{metric.name}",
                trace_id=trace_id,
                value=float(score)
            )

        results[i] = sample_scores

    # 全体の平均スコアを計算
    avg_scores = {}
    for metric_name in metrics[0].name:  # 最初のメトリクスの名前を使用
        scores = [sample_scores.get(metric_name, 0) for sample_scores in results.values()]
        avg_scores[metric_name] = sum(scores) / len(scores)
        logger.success(f"{model_name} - 平均{metric_name}: {avg_scores[metric_name]}")

    return avg_scores

async def main():
    logger.info("評価プロセスを開始")
    # 環境設定の読み込みと検証
    load_dotenv()
    check_required_env_vars()
    logger.info("環境設定を読み込み完了")
    
    # Langfuseのセットアップ
    langfuse = Langfuse()
    handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    logger.info("Langfuseの設定が完了")
    
    # データセットの準備
    dataset = load_mystery_dataset()
    eval_dataset = prepare_evaluation_dataset(dataset)
    
    # 評価対象のモデル
    models = ["gpt-4o-mini", "gpt-4o"]
    logger.info(f"評価対象モデル: {', '.join(models)}")
    
    for model_name in models:
        # 各モデルでの評価を実行
        llm = ChatOpenAI(model=model_name, temperature=0)
        embeddings = OpenAIEmbeddings()
        metrics = setup_ragas_evaluator(llm, embeddings)
        
        await evaluate_with_model(model_name, eval_dataset, metrics, langfuse, handler)
    
    logger.success("すべての評価プロセスが完了")

if __name__ == "__main__":
    asyncio.run(main())
