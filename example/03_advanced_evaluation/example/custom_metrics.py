#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional
from ragas.metrics.base import MetricWithLLM, SingleTurnMetric, MetricType, SingleTurnSample
from ragas.callbacks import Callbacks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dataclasses import dataclass, field
from ragas.metrics import Faithfulness

@dataclass
class MysteryRelevanceMetric(MetricWithLLM, SingleTurnMetric):
    """
    ミステリー小説特有の要素（動機、アリバイ、証拠など）に
    回答がどの程度言及しているかを評価するカスタムメトリクス
    """
    # name of the metric
    name: str = "mystery_relevance_metric"
    # required columns for the metric
    _required_columns: Dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    llm: Optional[ChatOpenAI] = None
    embeddings: Optional[OpenAIEmbeddings] = None

    def __post_init__(self):
        self.llm = self.llm or ChatOpenAI(temperature=0)
        self.embeddings = self.embeddings or OpenAIEmbeddings()

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks
    ) -> float:
        from langchain_core.messages import HumanMessage
        # 回答の事実性を評価するプロンプト
        faithfulness_prompt = """
        以下の回答が、与えられたコンテキストの情報のみに基づいているかを評価してください。
        
        コンテキスト:
        {context}
        
        回答:
        {answer}
        
        評価基準:
        1. 回答がコンテキストの情報のみに基づいているか
        2. コンテキストにない情報を追加していないか
        3. コンテキストの情報を正確に解釈しているか
        
        0（コンテキストと全く一致しない）から1（完全に一致する）の間で
        スコアを数値のみで出力してください。
        """
        
        context = "\n".join(sample.retrieved_contexts)
        message = HumanMessage(
            content=faithfulness_prompt.format(
                context=context,
                answer=sample.response
            )
        )
        score_response = await self.llm.ainvoke([message])
        try:
            base_score = float(score_response.content.strip())
        except ValueError:
            base_score = 0.0

        # マーダーミステリー要素を評価するために、キーワードの出現数に基づくボーナスを付与する
        keywords = ["murder", "detective", "clue", "suspect", "weapon", "crime"]
        response_text = sample.response.lower()
        found_count = sum(1 for word in keywords if word in response_text)
        bonus = min(found_count * 0.05, 0.3)
        score = min(base_score + bonus, 1.0)
        return score

@dataclass
class HallucinationsMetric(MetricWithLLM, SingleTurnMetric):
    # name of the metric
    name: str = "hallucinations_metric"
    # required columns for the metric
    _required_columns: Dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    llm: Optional[ChatOpenAI] = None
    embeddings: Optional[OpenAIEmbeddings] = None

    def __post_init__(self):
        self.llm = self.llm or ChatOpenAI(temperature=0)
        self.embeddings = self.embeddings or OpenAIEmbeddings()

    async def _single_turn_ascore(self, sample: SingleTurnSample, callbacks: Callbacks) -> float:
        from langchain_core.messages import HumanMessage
        # 回答の幻覚（ハルシネーション）を評価するプロンプト
        hallucination_prompt = """
        以下の回答に、与えられたコンテキストにない情報（幻覚）がどの程度含まれているかを評価してください。
        
        コンテキスト:
        {context}
        
        回答:
        {answer}
        
        評価基準:
        1. コンテキストにない情報や事実を追加していないか
        2. コンテキストの情報を誤って解釈・拡大解釈していないか
        3. 根拠のない推測や仮定を含んでいないか
        
        0（幻覚が全くない）から1（完全に幻覚）の間で
        スコアを数値のみで出力してください。
        """
        
        context = "\n".join(sample.retrieved_contexts)
        message = HumanMessage(
            content=hallucination_prompt.format(
                context=context,
                answer=sample.response
            )
        )
        score_response = await self.llm.ainvoke([message])
        try:
            score = float(score_response.content.strip())
            return score
        except ValueError:
            return 1.0  # エラーの場合は最悪のスコアを返す

@dataclass
class NarrativeCoherenceMetric(MetricWithLLM, SingleTurnMetric):
    """
    回答の論理的一貫性と説明の流れを評価するカスタムメトリクス
    """
    # name of the metric
    name: str = "narrative_coherence"
    # required columns for the metric
    _required_columns: Dict[MetricType, set[str]] = field(
        default_factory=lambda: {
            MetricType.SINGLE_TURN: {"user_input", "response", "retrieved_contexts"}
        }
    )
    llm: Optional[ChatOpenAI] = None

    def __post_init__(self):
        self.llm = self.llm or ChatOpenAI(temperature=0)

    async def _single_turn_ascore(
        self,
        sample: SingleTurnSample,
        callbacks: Callbacks
    ) -> float:
        from langchain_core.messages import HumanMessage
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
        
        message = HumanMessage(content=coherence_prompt.format(answer=sample.response))
        score_response = await self.llm.ainvoke([message])
        try:
            score = float(score_response.content.strip())
            return score
        except ValueError:
            return 0.0

# 使用例
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    async def test_metrics():
        load_dotenv()

        embeddings = OpenAIEmbeddings()

        # テスト用データ
        test_data = SingleTurnSample(
            user_input="深夜の美術館殺人事件の犯人は誰ですか？",
            response="""
            犯人は美術館の学芸員である山田健一です。

            その根拠として：
            1. 犯行時刻に美術館に入れる立場にあった
            2. 金庫の暗証番号を知っていた
            3. 防犯カメラの映像を消去できる権限があった
            4. 被害者との金銭トラブルという動機があった

            山田は金庫内の重要書類を盗むために警備員を殺害し、
            他の容疑者に疑いがかかるよう血で数列を書き残しました。
            """,
            retrieved_contexts=[
                "警備員の遺体は美術館2階で発見された",
                "展示室の床に謎めいた数列が血で書かれていた",
                "山田健一は被害者と金銭トラブルがあった"
            ],
            reference="""
            犯人は山田健一（美術館学芸員）です。
            被害者との金銭トラブルを抱えており、重要書類を盗むために警備員を殺害しました。
            犯行後、他の容疑者に疑いがかかるよう血で数列を書き残しました。
            """
        )

        models = ["gpt-4o-mini", "gpt-4o-mini"]

        for model in models:
            print(f"=== {model} での評価開始 ===")
            llm = ChatOpenAI(temperature=0, model=model)

            mystery_metric = MysteryRelevanceMetric(llm=llm, embeddings=embeddings)
            coherence_metric = NarrativeCoherenceMetric(llm=llm)
            hallucinations_metric = HallucinationsMetric(llm=llm, embeddings=embeddings)

            mystery_score = await mystery_metric._single_turn_ascore(test_data, None)
            coherence_score = await coherence_metric._single_turn_ascore(test_data, None)
            hallucinations_score = await hallucinations_metric._single_turn_ascore(test_data, None)

            print(f"Model: {model}")
            print("Mystery Relevance Score:", mystery_score)
            print("Narrative Coherence Score:", coherence_score)
            print("Hallucinations Score:", hallucinations_score)
            print("\n")

    asyncio.run(test_metrics())
