#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import asyncio
from dotenv import load_dotenv
from loguru import logger
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler

from metrics import setup_ragas_evaluator
from data_loader import (
    load_mystery_dataset,
    prepare_evaluation_dataset,
)
from evaluation import evaluate_with_model

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
        
        await evaluate_with_model(model_name, eval_dataset, metrics, langfuse, embeddings, llm)
    
    logger.success("すべての評価プロセスが完了")

if __name__ == "__main__":
    asyncio.run(main())
