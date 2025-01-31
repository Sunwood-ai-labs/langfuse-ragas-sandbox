#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional
from ragas.metrics.base import Metric
from ragas.metrics._utils import similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class MysteryRelevanceMetric(Metric):
    """
    ミステリー小説特有の要素（動機、アリバイ、証拠など）に
    回答がどの程度言及しているかを評価するカスタムメトリクス
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        self.llm = llm or ChatOpenAI(temperature=0)
        self.embeddings = embeddings or OpenAIEmbeddings()
        
        self.mystery_elements = [
            "動機", "アリバイ", "証拠", "時系列",
            "容疑者", "犯行手法", "トリック"
        ]
    
    async def _acompute_score(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        # 回答文中の各ミステリー要素への言及をチェック
        scores = []
        for element in self.mystery_elements:
            element_prompt = f"""
            以下の回答文が「{element}」についてどの程度言及しているか、
            0（全く言及なし）から1（十分な言及あり）の間でスコアを評価してください。

            回答文: {answer}

            スコアを数値のみで出力してください。
            """
            
            score_response = await self.llm.ainvoke(element_prompt)
            try:
                score = float(score_response.content.strip())
                scores.append(score)
            except ValueError:
                scores.append(0.0)
        
        # 全要素の平均スコアを算出
        final_score = sum(scores) / len(scores)
        
        return {
            "mystery_relevance_score": final_score,
            "element_scores": dict(zip(self.mystery_elements, scores))
        }
    
    @property
    def name(self) -> str:
        return "mystery_relevance"

class NarrativeCoherenceMetric(Metric):
    """
    回答の論理的一貫性と説明の流れを評価するカスタムメトリクス
    """
    
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None
    ):
        self.llm = llm or ChatOpenAI(temperature=0)
    
    async def _acompute_score(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        coherence_prompt = """
        以下の回答文の論理的一貫性と説明の流れを評価してください。
        
        評価基準:
        1. 説明の順序は論理的か
        2. 因果関係は明確か
        3. 矛盾する記述はないか
        4. 結論に至るまでの推論過程は妥当か
        
        回答文: {answer}
        
        0（全く一貫性がない）から1（完全に一貫している）の間で
        スコアを数値のみで出力してください。
        """
        
        score_response = await self.llm.ainvoke(coherence_prompt)
        try:
            score = float(score_response.content.strip())
        except ValueError:
            score = 0.0
        
        return {"narrative_coherence_score": score}
    
    @property
    def name(self) -> str:
        return "narrative_coherence"

# 使用例
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    async def test_metrics():
        load_dotenv()
        
        llm = ChatOpenAI(temperature=0)
        embeddings = OpenAIEmbeddings()
        
        # テスト用のデータ
        test_data = {
            "question": "深夜の美術館殺人事件の犯人は誰ですか？",
            "answer": """
            犯人は美術館の学芸員である山田健一です。
            
            その根拠として：
            1. 犯行時刻に美術館に入れる立場にあった
            2. 金庫の暗証番号を知っていた
            3. 防犯カメラの映像を消去できる権限があった
            4. 被害者との金銭トラブルという動機があった
            
            山田は金庫内の重要書類を盗むために警備員を殺害し、
            他の容疑者に疑いがかかるよう血で数列を書き残しました。
            """,
            "contexts": [
                "警備員の遺体は美術館2階で発見された",
                "展示室の床に謎めいた数列が血で書かれていた",
                "山田健一は被害者と金銭トラブルがあった"
            ]
        }
        
        # カスタムメトリクスのテスト
        mystery_metric = MysteryRelevanceMetric(llm, embeddings)
        coherence_metric = NarrativeCoherenceMetric(llm)
        
        mystery_score = await mystery_metric._acompute_score(**test_data)
        coherence_score = await coherence_metric._acompute_score(**test_data)
        
        print("Mystery Relevance Score:", mystery_score)
        print("Narrative Coherence Score:", coherence_score)
    
    asyncio.run(test_metrics())
