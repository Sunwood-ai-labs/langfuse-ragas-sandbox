#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from loguru import logger
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from tqdm import tqdm

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
