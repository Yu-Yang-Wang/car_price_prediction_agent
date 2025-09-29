"""增强的工作器，集成RAG功能提升分析质量"""

import os
import sys
from typing import Dict, Any, Optional
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .models import CarAnalysisState
from .workers import (
    price_research_worker as original_price_research_worker,
    price_comparison_worker as original_price_comparison_worker,
    deal_scoring_worker as original_deal_scoring_worker,
    llm_opinion_worker as original_llm_opinion_worker
)
from database.manager import DatabaseManager
from rag.rag_system import RAGSystem
from car_analysis.graph.graph_service import GraphService

logger = logging.getLogger(__name__)


class RAGEnhancedAnalysis:
    """RAG增强的分析器"""

    _instance = None
    _rag_system = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def rag_system(self):
        if self._rag_system is None:
            try:
                self._rag_system = RAGSystem()
                print("🧠 RAG system initialized for enhanced analysis")
            except Exception as e:
                logger.error(f"Failed to initialize RAG system: {e}")
                self._rag_system = None
        return self._rag_system


# 创建全局实例
rag_enhanced = RAGEnhancedAnalysis()


async def rag_enhanced_price_research_worker(state: CarAnalysisState) -> CarAnalysisState:
    """RAG增强的价格研究工作器"""
    print("🔍 Enhanced price research with RAG...")

    # 首先执行原始的价格研究
    result = await original_price_research_worker(state)

    # 如果RAG系统可用，添加增强分析
    if rag_enhanced.rag_system:
        try:
            car = state.get("current_car", {})

            # 查找相似车辆案例
            similar_cases = rag_enhanced.rag_system.find_similar_cases(car)

            if similar_cases.get('similar_cases'):
                print(f"   🔍 Found {len(similar_cases['similar_cases'])} similar cases for context")

                # 将相似案例信息添加到状态中
                result["rag_similar_cases"] = similar_cases['similar_cases']
                result["rag_cases_analysis"] = similar_cases.get('analysis', '')

                # 增强价格研究质量评分
                original_research = result.get("price_research", {})
                if original_research.get("success"):
                    original_research["rag_enhanced"] = True
                    original_research["rag_similarity_count"] = len(similar_cases['similar_cases'])

                    # 基于相似案例调整数据质量评分
                    if len(similar_cases['similar_cases']) >= 3:
                        original_research["research_quality"] = "high_with_rag"
                    elif len(similar_cases['similar_cases']) >= 1:
                        original_research["research_quality"] = "medium_with_rag"

        except Exception as e:
            logger.error(f"RAG enhancement failed in price research: {e}")
            # 不影响原始分析，继续流程

    return result


async def rag_enhanced_llm_opinion_worker(state: CarAnalysisState) -> CarAnalysisState:
    """RAG增强的LLM意见工作器"""
    print("🧠 Enhanced LLM opinion with RAG...")

    # 检查前置条件
    price_comparison = state.get("price_comparison", {})
    if not price_comparison.get("success"):
        print("   ⚠️ Price comparison not ready, skipping enhanced LLM opinion")
        return {"llm_opinion": {"success": False, "error": "Price comparison not available"}}

    car = state.get("current_car", {})
    rule_score = state.get("deal_score", {})

    # 如果RAG系统可用，使用RAG增强分析
    if rag_enhanced.rag_system:
        try:
            # 构建增强的分析上下文
            analysis_context = _build_analysis_context(state)

            # 使用RAG增强汽车分析
            rag_result = rag_enhanced.rag_system.enhance_car_analysis(
                car_data=car,
                analysis_context=analysis_context
            )

            enhanced_analysis = rag_result.get('enhanced_analysis', '')
            rag_confidence = rag_result.get('rag_confidence', 0.0)

            if enhanced_analysis:
                print(f"   🧠 RAG enhanced analysis generated (confidence: {rag_confidence:.2f})")

                # 尝试从增强分析中提取评分
                extracted_score = _extract_score_from_rag_analysis(enhanced_analysis)

                return {
                    "llm_opinion": {
                        "success": True,
                        "score": extracted_score.get('score', 50),
                        "verdict": extracted_score.get('verdict', 'Unknown'),
                        "reasoning": enhanced_analysis,
                        "rag_enhanced": True,
                        "rag_confidence": rag_confidence,
                        "rag_retrieval_info": rag_result.get('retrieved_info', '')
                    }
                }

        except Exception as e:
            logger.error(f"RAG enhancement failed in LLM opinion: {e}")
            # 降级到原始方法

    # 如果RAG失败或不可用，使用原始LLM工作器
    print("   🔄 Falling back to original LLM analysis")
    return await original_llm_opinion_worker(state)


async def rag_enhanced_deal_scoring_worker(state: CarAnalysisState) -> CarAnalysisState:
    """RAG增强的交易评分工作器"""
    print("🎯 Enhanced deal scoring with RAG...")

    # 首先执行原始评分
    result = await original_deal_scoring_worker(state)

    # 如果RAG系统可用且有相似案例，调整评分
    if rag_enhanced.rag_system and state.get("rag_similar_cases"):
        try:
            similar_cases = state.get("rag_similar_cases", [])
            original_score = result.get("deal_score", {})

            if original_score.get("success") and len(similar_cases) >= 2:
                # 计算相似案例的平均评分
                similar_scores = []
                for case in similar_cases:
                    car_info = case.get('car', {})
                    analysis = car_info.get('analysis', {})
                    if analysis.get('rule_based_score'):
                        similar_scores.append(analysis['rule_based_score'])

                if similar_scores:
                    avg_similar_score = sum(similar_scores) / len(similar_scores)
                    original_rule_score = original_score.get("score", 50)

                    # 混合原始评分和相似案例评分（权重：原始70%，相似案例30%）
                    rag_adjusted_score = int(original_rule_score * 0.7 + avg_similar_score * 0.3)

                    print(f"   🎯 RAG score adjustment: {original_rule_score} → {rag_adjusted_score} "
                          f"(based on {len(similar_scores)} similar cases)")

                    # 更新评分
                    original_score["rag_adjusted_score"] = rag_adjusted_score
                    original_score["rag_original_score"] = original_rule_score
                    original_score["rag_similar_cases_count"] = len(similar_scores)
                    original_score["rag_similar_avg_score"] = avg_similar_score

                    # 重新计算评价等级
                    new_verdict = _score_to_verdict(rag_adjusted_score)
                    original_score["rag_adjusted_verdict"] = new_verdict

        except Exception as e:
            logger.error(f"RAG score adjustment failed: {e}")

    return result


def _build_analysis_context(state: CarAnalysisState) -> str:
    """构建分析上下文"""
    context_parts = []

    # 价格比较信息
    price_comparison = state.get("price_comparison", {})
    if price_comparison.get("success"):
        market_price = price_comparison.get("market_median_price", 0)
        price_delta = price_comparison.get("price_delta", 0)
        context_parts.append(f"Market median price: ${market_price:,.0f}")
        context_parts.append(f"Price difference: ${price_delta:+,.0f}")

    # 规则评分信息
    deal_score = state.get("deal_score", {})
    if deal_score.get("success"):
        score = deal_score.get("score", 0)
        verdict = deal_score.get("verdict", "Unknown")
        context_parts.append(f"Rule-based score: {score}/100 ({verdict})")

    # 价格研究质量
    price_research = state.get("price_research", {})
    if price_research.get("success"):
        comparable_count = price_research.get("comparable_count", 0)
        data_source = price_research.get("data_source", "unknown")
        context_parts.append(f"Market data: {comparable_count} comparable vehicles from {data_source}")

    return ". ".join(context_parts)


def _extract_score_from_rag_analysis(analysis_text: str) -> Dict[str, Any]:
    """从RAG分析文本中提取评分信息"""
    try:
        # 简单的关键词提取（可以改进为更复杂的NLP）
        text_lower = analysis_text.lower()

        # 提取评分
        score = 50  # 默认评分
        if "excellent" in text_lower or "outstanding" in text_lower:
            score = 90
        elif "very good" in text_lower or "great deal" in text_lower:
            score = 85
        elif "good deal" in text_lower or "good" in text_lower:
            score = 75
        elif "fair" in text_lower or "reasonable" in text_lower:
            score = 60
        elif "poor" in text_lower or "bad deal" in text_lower:
            score = 40
        elif "terrible" in text_lower or "avoid" in text_lower:
            score = 25

        # 提取评价
        verdict = _score_to_verdict(score)

        return {"score": score, "verdict": verdict}

    except Exception as e:
        logger.error(f"Error extracting score from RAG analysis: {e}")
        return {"score": 50, "verdict": "Unknown"}


def _score_to_verdict(score: int) -> str:
    """将评分转换为评价等级"""
    if score >= 90:
        return "Exceptional Deal ⭐⭐⭐"
    elif score >= 80:
        return "Good Deal ⭐⭐"
    elif score >= 60:
        return "Fair Deal ⭐"
    elif score >= 40:
        return "Poor Deal ⚠️"
    else:
        return "Bad Deal ❌"


# =============== 数据持久化工作器 ===============

async def save_analysis_to_database(state: CarAnalysisState) -> CarAnalysisState:
    """保存分析结果到数据库"""
    if not rag_enhanced.rag_system:
        print("   ⚠️ RAG system not available, skipping database save")
        return state

    try:
        car = state.get("current_car", {})

        # 保存汽车数据
        car_id = rag_enhanced.rag_system.db_manager.save_car(
            car_data=car,
            session_id=state.get("session_id")
        )

        # 收集分析数据
        analysis_data = {
            "rule_based_score": state.get("deal_score", {}).get("score"),
            "rule_based_verdict": state.get("deal_score", {}).get("verdict"),
            "llm_score": state.get("llm_opinion", {}).get("score"),
            "llm_verdict": state.get("llm_opinion", {}).get("verdict"),
            "llm_reasoning": state.get("llm_opinion", {}).get("reasoning"),
            "market_median_price": state.get("price_comparison", {}).get("market_median_price"),
            "price_delta": state.get("price_comparison", {}).get("price_delta"),
            "price_delta_percent": state.get("price_comparison", {}).get("price_delta_percent"),
            "deal_category": state.get("price_comparison", {}).get("verdict"),
            "data_source": state.get("price_research", {}).get("data_source"),
            "comparable_count": state.get("price_research", {}).get("comparable_count"),
            "research_quality": state.get("price_research", {}).get("research_quality"),
            "success": state.get("deal_score", {}).get("success", False),
            "analysis_version": "2.0_rag_enhanced"
        }

        # 保存分析结果
        analysis_id = rag_enhanced.rag_system.db_manager.save_analysis(car_id, analysis_data)

        # 保存市场数据（如果有）
        market_data = state.get("price_research", {}).get("comparable_prices", [])
        if market_data:
            # 转换格式
            formatted_market_data = []
            for price_info in market_data:
                formatted_data = {
                    "search_query": f"{car.get('year')} {car.get('make')} {car.get('model')}",
                    "price": price_info.get("price", 0),
                    "url": price_info.get("url", ""),
                    "source": price_info.get("source", "tavily"),
                    "similarity_score": 1.0
                }
                formatted_market_data.append(formatted_data)

            rag_enhanced.rag_system.db_manager.save_market_data(car_id, formatted_market_data)

        # 同步到向量存储
        rag_enhanced.rag_system.sync_car_to_vector_store(car_id)
        rag_enhanced.rag_system.sync_analysis_to_vector_store(analysis_id, car_id)

        # 同步到图数据库（可选）
        try:
            graph = getattr(rag_enhanced.rag_system, "graph_service", None)
            if graph and isinstance(graph, GraphService) and graph.available:
                # 使用现有 car/analysis 数据进行 upsert
                graph.upsert_car(car_id, car)
                graph.upsert_analysis(analysis_id, analysis_data)
                graph.link_car_analysis(car_id, analysis_id)
                print("   🕸️ Synced nodes/edges to graph")
        except Exception as ge:
            logger.warning(f"Graph sync skipped: {ge}")

        print(f"   ✅ Analysis saved to database (Car ID: {car_id}, Analysis ID: {analysis_id})")

        return {
            **state,
            "database_car_id": car_id,
            "database_analysis_id": analysis_id,
            "database_saved": True
        }

    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return {
            **state,
            "database_saved": False,
            "database_error": str(e)
        }


# =============== 导出增强的工作器 ===============

# 为了保持向后兼容性，也导出原始工作器
__all__ = [
    # RAG增强的工作器
    'rag_enhanced_price_research_worker',
    'rag_enhanced_llm_opinion_worker',
    'rag_enhanced_deal_scoring_worker',
    'save_analysis_to_database',

    # 原始工作器（向后兼容）
    'price_research_worker',
    'price_comparison_worker',
    'deal_scoring_worker',
    'llm_opinion_worker'
]

# 向后兼容的别名
price_research_worker = rag_enhanced_price_research_worker
price_comparison_worker = original_price_comparison_worker
deal_scoring_worker = rag_enhanced_deal_scoring_worker
llm_opinion_worker = rag_enhanced_llm_opinion_worker
