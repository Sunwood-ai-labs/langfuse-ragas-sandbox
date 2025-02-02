#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from loguru import logger
from langfuse.callback import CallbackHandler
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# ログの設定
logger.add(
    "logs/rag_app_{time}.log",
    rotation="500 MB",
    level="INFO",
    format="{time} {level} {message}",
)

def load_mystery_dataset() -> Dict:
    """ミステリーデータセットを読み込む"""
    logger.info("ミステリーデータセットの読み込みを開始")
    with open("../data/mystery_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"データセット読み込み完了: {len(data['cases'])}件の事件を読み込みました")
    return data

def create_documents(dataset: Dict) -> List[Document]:
    """データセットからドキュメントを作成"""
    logger.info("ドキュメントの作成を開始")
    documents = []
    for case in tqdm(dataset["cases"], desc="事件データの処理"):
        # 事件の説明文
        documents.append(Document(
            page_content=f"事件タイトル: {case['title']}\n説明: {case['description']}",
            metadata={"case_id": case["id"], "type": "description"}
        ))
        
        # 証拠品
        evidence_text = "\n".join([f"- {e}" for e in case["evidence"]])
        documents.append(Document(
            page_content=f"証拠品リスト:\n{evidence_text}",
            metadata={"case_id": case["id"], "type": "evidence"}
        ))
        
        # 容疑者情報
        for suspect in case["suspects"]:
            suspect_text = f"容疑者名: {suspect['name']}\n"
            suspect_text += f"役割: {suspect['role']}\n"
            suspect_text += f"アリバイ: {suspect['alibi']}\n"
            suspect_text += f"動機: {suspect['motive']}"
            documents.append(Document(
                page_content=suspect_text,
                metadata={"case_id": case["id"], "type": "suspect"}
            ))
    logger.info(f"ドキュメント作成完了: 合計{len(documents)}件のドキュメントを作成")
    return documents

def setup_callback_handler():
    """Langfuseのコールバックハンドラーのセットアップ"""
    logger.info("Langfuseコールバックハンドラーの設定を開始")
    load_dotenv()
    handler = CallbackHandler(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST")
    )
    logger.info("Langfuseコールバックハンドラーの設定完了")
    return handler

def create_rag_chain(documents: List[Document], model_name: str):
    """RAGチェーンの作成"""
    logger.info(f"RAGチェーンの作成を開始: モデル {model_name}")

    # テキストを分割
    logger.info("テキスト分割の処理を開始")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(documents)
    logger.info(f"テキスト分割完了: {len(splits)}個のチャンクを生成")
    
    # ベクターストアの作成
    logger.info("ベクターストアの作成を開始")
    embeddings = OpenAIEmbeddings()
    with tqdm(total=len(splits), desc="ベクトル化の進捗") as pbar:
        vectorstore = FAISS.from_documents(splits, embeddings)
        pbar.update(len(splits))
    retriever = vectorstore.as_retriever()
    logger.info("ベクターストア作成完了")
    
    # プロンプトの作成
    template = """以下の背景情報をもとに、質問に回答してください。

背景情報:
{context}

質問: {question}

回答は日本語で、以下の点に注意して具体的に説明してください：
1. 事実と推測を明確に区別する
2. 証拠に基づいた論理的な説明を心がける
3. ミステリーの要素（動機、アリバイ、証拠など）に着目する
4. 結論に至るまでの推論過程を明確にする"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # LLMの設定
    llm = ChatOpenAI(model=model_name, temperature=0)
    
    # チェーンの作成（RunnablePassthroughを使用）
    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    
    logger.info("RAGチェーンの作成完了")
    return chain

def main():
    try:
        logger.info("RAGアプリケーションの実行を開始")
        
        # Langfuseハンドラーの初期設定
        handler = setup_callback_handler()
        
        # データセットの読み込みとドキュメント作成
        dataset = load_mystery_dataset()
        documents = create_documents(dataset)
        
        # テスト質問
        questions = [
            "深夜の美術館殺人事件の概要を教えてください",
            "仮面舞踏会の事件で発見された証拠品は何ですか？",
            "美術館殺人事件の容疑者は誰がいますか？"
        ]
        
        # 使用するモデル
        models = ["gpt-4o", "gpt-4o-mini"]
        
        for model_name in models:
            logger.info(f"{model_name}での評価を開始")
            
            # RAGチェーンの作成
            chain = create_rag_chain(documents, model_name)
            
            for question in tqdm(questions, desc=f"{model_name}での質問処理"):
                logger.info(f"質問: {question}")
                response = chain.invoke(
                    question,
                    config={"callbacks": [handler]}
                )
                logger.info(f"回答 ({model_name}): {response.content}")
                print(f"\n質問: {question}")
                print(f"回答 ({model_name}): {response.content}\n")

        logger.info("RAGアプリケーションの実行が正常に完了")

    except Exception as e:
        logger.error(f"エラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
