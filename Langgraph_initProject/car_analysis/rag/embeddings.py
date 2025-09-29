"""Embedding管理器，用于生成和管理文本嵌入

增强点：
- 支持通过环境变量配置提供方与模型：
  - `EMBEDDINGS_PROVIDER` = "openai" | "huggingface"（缺省自动选择）
  - `OPENAI_EMBEDDING_MODEL`（缺省: text-embedding-ada-002）
  - `HF_EMBEDDING_MODEL`（缺省: sentence-transformers/all-MiniLM-L6-v2）
- 公开 `provider`、`model_id`、`embedding_dim` 供向量库做版本化集合命名
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """嵌入管理器"""

    def __init__(self,
                 model_name: Optional[str] = None,
                 openai_model: Optional[str] = None,
                 huggingface_model: Optional[str] = None):
        """初始化嵌入管理器

        Args:
            model_name: 模型名称 ("openai" 或 "huggingface")
            openai_model: OpenAI嵌入模型名称
            huggingface_model: HuggingFace嵌入模型名称
        """
        # 解析配置优先级：
        # 1) 显式传参 model_name
        # 2) 环境变量 EMBEDDINGS_PROVIDER
        # 3) 自动：如果有 OPENAI_API_KEY 用 openai，否则 huggingface

        env_provider = os.getenv("EMBEDDINGS_PROVIDER")
        provider = (model_name or env_provider or "auto").lower()

        # 读取模型名，支持环境变量覆盖
        openai_model = openai_model or os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-ada-002"
        huggingface_model = huggingface_model or os.getenv("HF_EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"

        if provider == "auto":
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "huggingface"

        self.provider = provider  # openai | huggingface
        self.model_id = None      # 具体的模型标识字符串

        if provider == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("EMBEDDINGS_PROVIDER=openai 但未发现 OPENAI_API_KEY，回退至 HuggingFace")
                self._init_huggingface(huggingface_model)
            else:
                self._init_openai(openai_model)
        else:
            self._init_huggingface(huggingface_model)

    def _init_openai(self, model: str):
        """初始化OpenAI嵌入"""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=model,
                chunk_size=1000
            )
            self.embedding_dim = 1536  # text-embedding-ada-002 dimension
            self.provider = "openai"
            self.model_id = model
            print(f"🤖 Initialized OpenAI embeddings: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            # 降级到HuggingFace
            self._init_huggingface("sentence-transformers/all-MiniLM-L6-v2")

    def _init_huggingface(self, model: str):
        """初始化HuggingFace嵌入"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
            self.provider = "huggingface"
            self.model_id = model
            print(f"🤗 Initialized HuggingFace embeddings: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """为单个文本生成嵌入

        Args:
            text: 输入文本

        Returns:
            嵌入向量
        """
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim

            embedding = self.embeddings.embed_query(text)
            return embedding

        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            return [0.0] * self.embedding_dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """为多个文本生成嵌入

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        try:
            # 过滤空文本
            valid_texts = [text if text and text.strip() else " " for text in texts]
            embeddings = self.embeddings.embed_documents(valid_texts)
            return embeddings

        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return [[0.0] * self.embedding_dim] * len(texts)

    def calculate_similarity(self,
                           embedding1: List[float],
                           embedding2: List[float]) -> float:
        """计算两个嵌入向量的余弦相似度

        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量

        Returns:
            余弦相似度 (-1 到 1)
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def create_car_description(self, car_data: Dict[str, Any]) -> str:
        """为汽车数据创建描述性文本，用于嵌入

        Args:
            car_data: 汽车数据字典

        Returns:
            描述性文本
        """
        try:
            year = car_data.get('year', '')
            make = car_data.get('make', '')
            model = car_data.get('model', '')
            mileage = car_data.get('mileage', 0)
            price = car_data.get('price_paid', 0)
            trim = car_data.get('trim', '')
            color = car_data.get('color', '')
            transmission = car_data.get('transmission', '')
            engine = car_data.get('engine', '')
            fuel_type = car_data.get('fuel_type', '')
            condition = car_data.get('condition', '')
            location = car_data.get('location', '')

            # 构建描述性文本
            parts = []

            # 基本信息
            if year and make and model:
                basic = f"{year} {make} {model}"
                if trim:
                    basic += f" {trim}"
                parts.append(basic)

            # 里程和价格
            if mileage:
                parts.append(f"{mileage:,} miles")
            if price:
                parts.append(f"${price:,.0f}")

            # 详细信息
            details = []
            if color:
                details.append(f"{color} color")
            if transmission:
                details.append(f"{transmission} transmission")
            if engine:
                details.append(f"{engine} engine")
            if fuel_type:
                details.append(f"{fuel_type} fuel")
            if condition:
                details.append(f"{condition} condition")
            if location:
                details.append(f"located in {location}")

            if details:
                parts.append(", ".join(details))

            description = ". ".join(parts)

            # 添加原始文本（如果有）
            raw_text = car_data.get('raw_text', '')
            if raw_text and len(raw_text.strip()) > 0:
                description += f". Additional details: {raw_text[:200]}"

            return description

        except Exception as e:
            logger.error(f"Error creating car description: {e}")
            return f"{car_data.get('year', '')} {car_data.get('make', '')} {car_data.get('model', '')}"

    def create_analysis_description(self, analysis_data: Dict[str, Any]) -> str:
        """为分析结果创建描述性文本

        Args:
            analysis_data: 分析数据字典

        Returns:
            描述性文本
        """
        try:
            parts = []

            # 评分信息
            rule_score = analysis_data.get('rule_based_score')
            rule_verdict = analysis_data.get('rule_based_verdict')
            llm_score = analysis_data.get('llm_score')
            llm_verdict = analysis_data.get('llm_verdict')

            if rule_score is not None and rule_verdict:
                parts.append(f"Rule-based analysis: {rule_score}/100 ({rule_verdict})")

            if llm_score is not None and llm_verdict:
                parts.append(f"LLM analysis: {llm_score}/100 ({llm_verdict})")

            # 价格分析
            market_price = analysis_data.get('market_median_price')
            price_delta = analysis_data.get('price_delta')
            price_delta_percent = analysis_data.get('price_delta_percent')

            if market_price is not None:
                parts.append(f"Market median price: ${market_price:,.0f}")

            if price_delta is not None and price_delta_percent is not None:
                direction = "above" if price_delta > 0 else "below"
                parts.append(f"Price is ${abs(price_delta):,.0f} ({abs(price_delta_percent):.1f}%) {direction} market")

            # 数据质量
            comparable_count = analysis_data.get('comparable_count')
            data_source = analysis_data.get('data_source')

            if comparable_count:
                parts.append(f"Based on {comparable_count} comparable vehicles")

            if data_source:
                parts.append(f"Data source: {data_source}")

            # LLM推理
            llm_reasoning = analysis_data.get('llm_reasoning', '')
            if llm_reasoning:
                parts.append(f"Analysis reasoning: {llm_reasoning[:300]}")

            return ". ".join(parts)

        except Exception as e:
            logger.error(f"Error creating analysis description: {e}")
            return "Analysis data available"

    def create_knowledge_text(self, knowledge_data: Dict[str, Any]) -> str:
        """为知识库条目创建完整文本

        Args:
            knowledge_data: 知识数据字典

        Returns:
            完整文本
        """
        try:
            title = knowledge_data.get('title', '')
            content = knowledge_data.get('content', '')
            category = knowledge_data.get('category', '')
            tags = knowledge_data.get('tags', [])

            parts = []

            if title:
                parts.append(f"Title: {title}")

            if category:
                parts.append(f"Category: {category}")

            if tags:
                parts.append(f"Tags: {', '.join(tags)}")

            if content:
                parts.append(f"Content: {content}")

            return ". ".join(parts)

        except Exception as e:
            logger.error(f"Error creating knowledge text: {e}")
            return knowledge_data.get('content', '')

    def get_embedding_info(self) -> Dict[str, Any]:
        """获取嵌入模型信息

        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'model_type': type(self.embeddings).__name__
        }
