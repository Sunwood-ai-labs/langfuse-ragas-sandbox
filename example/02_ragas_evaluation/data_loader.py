#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict
from loguru import logger
from datasets import Dataset
from tqdm import tqdm

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
