#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Dict, Any
import uuid
from loguru import logger
from tqdm import tqdm
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from ragas.dataset_schema import SingleTurnSample

async def evaluate_with_model(model_name: str, eval_dataset: Dataset, metrics: List, langfuse: Any, embeddings: Any):
    """指定されたモデルで評価を実行"""
    logger.info(f"{model_name}による評価を開始")

    # RAGチェーンの設定
    prompt = ChatPromptTemplate.from_template(
        "背景情報をもとに質問に回答してください。背景情報： {context} 質問： {question}"
    )

    results = {}
    for i, sample in enumerate(tqdm(eval_dataset, desc=f"{model_name}の評価進捗")):
        # トレースの作成
        trace = langfuse.trace(
            name=f"RAG Evaluation - {model_name}",
            metadata={
                "model": model_name,
                "question": sample["question"],
                "ground_truth": sample["ground_truths"][0]
            }
        )

        with trace:
            # 検索の実行
            vectorstore = FAISS.from_texts(texts=sample["contexts"], embedding=embeddings)
            retriever = vectorstore.as_retriever()
            
            # 検索結果をスパンとして記録
            retrieval_span = trace.span(
                name="retrieval",
                input={"question": sample["question"]},
            )
            retrieved_docs = retriever.get_relevant_documents(sample["question"])
            retrieval_span.end(
                output={"contexts": [doc.page_content for doc in retrieved_docs]}
            )

            # 回答生成チェーンの設定と実行
            chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            # 生成をスパンとして記録
            generation_span = trace.span(
                name="generation",
                input={
                    "question": sample["question"],
                    "contexts": sample["contexts"]
                }
            )
            answer = await chain.ainvoke(
                sample["question"]
            )
            generation_span.end(output={"answer": answer})

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
                trace.score(
                    name=metric.name,
                    value=float(score)
                )

            results[i] = sample_scores

    # 全体の平均スコアを計算
    avg_scores = {}
    # 各メトリクスの名前を使用して平均を計算
    metric_names = [metric.name for metric in metrics]
    for metric_name in metric_names:
        scores = [sample_scores.get(metric_name, 0) for sample_scores in results.values()]
        if scores:  # スコアが存在する場合のみ平均を計算
            avg_scores[metric_name] = sum(scores) / len(scores)
            logger.success(f"{model_name} - 平均{metric_name}: {avg_scores[metric_name]}")

    return avg_scores
