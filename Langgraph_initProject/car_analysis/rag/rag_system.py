"""RAG (Retrieval-Augmented Generation) 系统核心实现"""

import os
import sys
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .vector_store import VectorStoreManager
from .embeddings import EmbeddingManager
from database.manager import DatabaseManager
from car_analysis.graph.graph_service import GraphService

logger = logging.getLogger(__name__)


class RAGSystem:
    """RAG系统主类"""

    def __init__(self,
                 db_manager: Optional[DatabaseManager] = None,
                 vector_manager: Optional[VectorStoreManager] = None,
                 embedding_manager: Optional[EmbeddingManager] = None,
                 llm_model: str = "gpt-4o",
                 temperature: float = 0.3):
        """初始化RAG系统

        Args:
            db_manager: 数据库管理器
            vector_manager: 向量存储管理器
            embedding_manager: 嵌入管理器
            llm_model: LLM模型名称
            temperature: LLM温度参数
        """
        # 初始化组件
        self.db_manager = db_manager or DatabaseManager()
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vector_manager = vector_manager or VectorStoreManager(
            embedding_manager=self.embedding_manager
        )

        # 初始化LLM
        try:
            self.llm = ChatOpenAI(
                model=llm_model,
                temperature=temperature
            )
            print(f"🤖 RAG System initialized with {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

        # 创建提示模板
        self._create_prompt_templates()

        # 初始化图服务（可选）
        try:
            self.graph_service = GraphService()
            if not self.graph_service.available:
                self.graph_service = None
                print("ℹ️ GraphService not available; continuing without GraphRAG")
            else:
                print("🕸️ GraphRAG enabled")
        except Exception as ge:  # pragma: no cover
            logger.warning(f"GraphService init failed: {ge}")
            self.graph_service = None

    def _create_prompt_templates(self):
        """创建各种提示模板"""

        # 汽车分析增强提示
        self.car_analysis_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
你是一位专业的汽车分析专家，具有丰富的汽车市场知识和价格评估经验。

你有access to以下相关信息：
{retrieved_context}

请基于提供的上下文信息和你的专业知识，对以下汽车进行深入分析。

分析要求：
1. 综合考虑市场数据、类似车辆分析、历史价格趋势
2. 提供专业的价格评估和交易建议
3. 解释你的推理过程
4. 如果发现矛盾或不一致的信息，请指出并给出合理解释
5. 考虑车辆的品牌可靠性、保值率、维修成本等因素

请用清晰、专业但易懂的语言回复。
"""),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        # 知识问答模板
        self.qa_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
你是一位汽车专家，可以回答关于汽车购买、市场分析、价格评估等各种问题。

相关参考信息：
{retrieved_context}

请基于提供的信息和你的专业知识回答用户的问题。如果信息不足，请说明需要更多信息。
"""),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        # 相似案例分析模板
        self.similar_cases_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("""
你是一位汽车价格分析专家。用户询问关于特定车辆的信息，你需要基于相似车辆的历史分析来提供洞察。

相似案例分析：
{retrieved_context}

请分析这些相似案例，总结规律和趋势，为用户提供有价值的参考信息。
"""),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

    def enhance_car_analysis(self,
                           car_data: Dict[str, Any],
                           analysis_context: str = None) -> Dict[str, Any]:
        """使用RAG增强汽车分析

        Args:
            car_data: 汽车数据
            analysis_context: 分析上下文

        Returns:
            增强的分析结果
        """
        try:
            # 1. 检索相关信息（GraphRAG 子图 + 向量检索融合）
            retrieved_info = self._retrieve_for_car_analysis(car_data)

            # 2. 构建查询
            car_description = self.embedding_manager.create_car_description(car_data)
            query = f"""
请分析以下车辆：
{car_description}

已知信息：
- 购买价格: ${car_data.get('price_paid', 0):,.0f}
- 里程数: {car_data.get('mileage', 0):,} miles

{analysis_context if analysis_context else ''}

请提供详细的价格分析和购买建议。
"""

            # 3. 生成增强回复
            enhanced_response = self._generate_response(
                template=self.car_analysis_template,
                query=query,
                retrieved_context=retrieved_info
            )

            return {
                "enhanced_analysis": enhanced_response,
                "retrieved_info": retrieved_info,
                "rag_confidence": self._calculate_confidence(retrieved_info)
            }

        except Exception as e:
            logger.error(f"Error in enhance_car_analysis: {e}")
            return {
                "enhanced_analysis": "无法生成增强分析",
                "error": str(e)
            }

    def answer_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """使用RAG回答用户问题

        Args:
            question: 用户问题
            context: 额外上下文

        Returns:
            回答结果
        """
        try:
            # 1. 检索相关信息
            retrieved_info = self._retrieve_for_question(question)

            # 2. 生成回答
            answer = self._generate_response(
                template=self.qa_template,
                query=question,
                retrieved_context=retrieved_info
            )

            # 3. 保存用户查询（用于改进系统）
            self._save_user_query(question, retrieved_info, answer, context)

            return {
                "answer": answer,
                "retrieved_info": retrieved_info,
                "confidence": self._calculate_confidence(retrieved_info)
            }

        except Exception as e:
            logger.error(f"Error in answer_question: {e}")
            return {
                "answer": "抱歉，无法回答这个问题。",
                "error": str(e)
            }

    def find_similar_cases(self,
                          car_data: Dict[str, Any],
                          analysis_type: str = "all") -> Dict[str, Any]:
        """查找相似案例并生成分析

        Args:
            car_data: 汽车数据
            analysis_type: 分析类型

        Returns:
            相似案例分析
        """
        try:
            # 1. 查找相似汽车
            similar_cars = self.vector_manager.search_similar_cars(
                query_car=car_data,
                limit=10,
                similarity_threshold=0.7
            )

            # 2. 获取相似汽车的分析结果
            similar_analyses = []
            for car in similar_cars:
                car_id = car['metadata'].get('car_id')
                if car_id:
                    car_with_analysis = self.db_manager.get_car_with_analysis(car_id)
                    if car_with_analysis and car_with_analysis.get('analysis'):
                        similar_analyses.append({
                            'car': car_with_analysis,
                            'similarity': car['similarity']
                        })

            # 3. 构建查询
            car_description = self.embedding_manager.create_car_description(car_data)
            query = f"""
用户询问关于这辆车的信息：
{car_description}

请基于相似案例分析，提供市场洞察和建议。
"""

            # 4. 格式化相似案例信息
            cases_context = self._format_similar_cases(similar_analyses)

            # 5. 生成分析
            analysis = self._generate_response(
                template=self.similar_cases_template,
                query=query,
                retrieved_context=cases_context
            )

            return {
                "analysis": analysis,
                "similar_cases": similar_analyses,
                "cases_count": len(similar_analyses)
            }

        except Exception as e:
            logger.error(f"Error in find_similar_cases: {e}")
            return {
                "analysis": "无法找到相似案例",
                "error": str(e)
            }

    def _retrieve_for_car_analysis(self, car_data: Dict[str, Any]) -> str:
        """为汽车分析检索相关信息"""
        try:
            # 构建搜索查询
            make = car_data.get('make', '')
            model = car_data.get('model', '')
            year = car_data.get('year', 0)

            search_queries = [
                f"{year} {make} {model} price analysis",
                f"{make} {model} market value",
                f"{make} reliability review",
                f"used car {make} {model} buying guide"
            ]

            retrieved_items = []

            # a) GraphRAG: 子图上下文（可选）
            graph_context = ""
            if getattr(self, "graph_service", None):
                try:
                    graph_context = self.graph_service.context_for_car(car_data) or ""
                except Exception as e:
                    logger.warning(f"Graph context retrieval failed: {e}")

            for query in search_queries:
                # 搜索知识库
                knowledge_results = self.vector_manager.search_knowledge(
                    query_text=query,
                    limit=3
                )

                # 搜索分析结果
                analysis_results = self.vector_manager.search_similar_analyses(
                    query_text=query,
                    limit=3
                )

                retrieved_items.extend(knowledge_results)
                retrieved_items.extend(analysis_results)

            # 格式化检索结果 + 拼接图上下文
            vect_text = self._format_retrieved_info(retrieved_items)
            if graph_context:
                return f"[Graph Context]\n{graph_context}\n\n[Vector Context]\n{vect_text}"
            return vect_text

        except Exception as e:
            logger.error(f"Error retrieving for car analysis: {e}")
            return "无可用参考信息"

    def _retrieve_for_question(self, question: str) -> str:
        """为问题回答检索相关信息"""
        try:
            # 跨集合搜索
            search_results = self.vector_manager.semantic_search(
                query_text=question,
                collections=["knowledge", "analyses", "cars"],
                limit=5
            )

            retrieved_items = []
            for collection, items in search_results.items():
                retrieved_items.extend(items)

            # 按相似度排序
            retrieved_items.sort(key=lambda x: x['similarity'], reverse=True)

            # 格式化检索结果
            return self._format_retrieved_info(retrieved_items[:10])

        except Exception as e:
            logger.error(f"Error retrieving for question: {e}")
            return "无可用参考信息"

    def _generate_response(self,
                          template: ChatPromptTemplate,
                          query: str,
                          retrieved_context: str) -> str:
        """生成RAG回复"""
        try:
            # 构建链
            chain = template | self.llm | StrOutputParser()

            # 生成回复
            response = chain.invoke({
                "query": query,
                "retrieved_context": retrieved_context
            })

            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "抱歉，生成回复时出现错误。"

    def _format_retrieved_info(self, retrieved_items: List[Dict[str, Any]]) -> str:
        """格式化检索到的信息"""
        if not retrieved_items:
            return "无相关参考信息"

        formatted_parts = []
        for i, item in enumerate(retrieved_items, 1):
            metadata = item.get('metadata', {})
            document = item.get('document', '')
            similarity = item.get('similarity', 0)

            # 根据不同类型格式化
            item_type = metadata.get('type', 'unknown')

            if item_type == 'knowledge':
                title = metadata.get('title', f'参考信息 {i}')
                formatted_parts.append(f"【{title}】(相似度: {similarity:.2f})\n{document}\n")

            elif item_type == 'analysis_result':
                car_id = metadata.get('car_id', 'unknown')
                score = metadata.get('rule_based_score', 'N/A')
                formatted_parts.append(f"【分析案例 {car_id}】(相似度: {similarity:.2f}, 评分: {score})\n{document}\n")

            elif item_type == 'car_data':
                make = metadata.get('make', '')
                model = metadata.get('model', '')
                year = metadata.get('year', '')
                formatted_parts.append(f"【{year} {make} {model}】(相似度: {similarity:.2f})\n{document}\n")

            else:
                formatted_parts.append(f"【参考信息 {i}】(相似度: {similarity:.2f})\n{document}\n")

        return "\n".join(formatted_parts)

    def _format_similar_cases(self, similar_analyses: List[Dict[str, Any]]) -> str:
        """格式化相似案例"""
        if not similar_analyses:
            return "无相似案例"

        formatted_cases = []
        for i, case in enumerate(similar_analyses, 1):
            car_info = case['car']
            analysis = car_info.get('analysis', {})
            similarity = case['similarity']

            car_desc = f"{car_info.get('year')} {car_info.get('make')} {car_info.get('model')}"
            price_paid = car_info.get('price_paid', 0)
            rule_score = analysis.get('rule_based_score', 'N/A')
            llm_score = analysis.get('llm_score', 'N/A')
            verdict = analysis.get('rule_based_verdict', 'N/A')

            case_text = f"""
案例 {i}: {car_desc} (相似度: {similarity:.2f})
- 成交价: ${price_paid:,.0f}
- 规则评分: {rule_score}/100
- LLM评分: {llm_score}/100
- 评价: {verdict}
- 推理: {analysis.get('llm_reasoning', 'N/A')[:200]}...
"""
            formatted_cases.append(case_text)

        return "\n".join(formatted_cases)

    def _calculate_confidence(self, retrieved_info: str) -> float:
        """计算回复的置信度"""
        try:
            # 基于检索信息的质量计算置信度
            if not retrieved_info or retrieved_info == "无相关参考信息":
                return 0.1

            # 简单的置信度计算（可以改进）
            lines = retrieved_info.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]

            if len(non_empty_lines) >= 10:
                return 0.9
            elif len(non_empty_lines) >= 5:
                return 0.7
            elif len(non_empty_lines) >= 2:
                return 0.5
            else:
                return 0.3

        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.1

    def _save_user_query(self,
                        query: str,
                        retrieved_info: str,
                        response: str,
                        context: Dict[str, Any] = None):
        """保存用户查询以改进系统"""
        try:
            # 这里可以保存到数据库用于分析和改进
            # 暂时只记录日志
            logger.info(f"User query logged: {query[:100]}...")

        except Exception as e:
            logger.error(f"Error saving user query: {e}")

    # =============== 数据同步方法 ===============

    def sync_car_to_vector_store(self, car_id: int) -> bool:
        """将汽车数据同步到向量存储"""
        try:
            car_data = self.db_manager.get_car(car_id)
            if car_data:
                return self.vector_manager.add_car(car_id, car_data)
            return False

        except Exception as e:
            logger.error(f"Error syncing car to vector store: {e}")
            return False

    def sync_analysis_to_vector_store(self, analysis_id: int, car_id: int) -> bool:
        """将分析结果同步到向量存储"""
        try:
            car_with_analysis = self.db_manager.get_car_with_analysis(car_id)
            if car_with_analysis and car_with_analysis.get('analysis'):
                analysis_data = car_with_analysis['analysis']
                return self.vector_manager.add_analysis(analysis_id, car_id, analysis_data)
            return False

        except Exception as e:
            logger.error(f"Error syncing analysis to vector store: {e}")
            return False

    def get_system_stats(self) -> Dict[str, Any]:
        """获取RAG系统统计信息"""
        try:
            db_stats = self.db_manager.get_stats()
            vector_stats = self.vector_manager.get_collection_stats()

            return {
                "database_stats": db_stats,
                "vector_store_stats": vector_stats,
                "embedding_info": self.embedding_manager.get_embedding_info(),
                "rag_system_status": "healthy"
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}
