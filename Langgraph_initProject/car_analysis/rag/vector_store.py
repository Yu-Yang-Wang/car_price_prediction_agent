"""向量存储管理器，使用Chroma作为向量数据库

增强点：
- 根据嵌入提供方/模型/维度对集合进行版本化命名，避免维度不匹配
  逻辑名 -> 物理名：`<logical>__<provider>_<model>_<dim>`
  例如：`knowledge__huggingface_all-minilm-l6-v2_384`
"""

import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import logging

from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self,
                 persist_directory: str = "database/chroma_db",
                 embedding_manager: Optional[EmbeddingManager] = None):
        """初始化向量存储管理器

        Args:
            persist_directory: 数据持久化目录
            embedding_manager: 嵌入管理器实例
        """
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化嵌入管理器
        self.embedding_manager = embedding_manager or EmbeddingManager()

        # 初始化Chroma客户端
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            print(f"📚 Vector store initialized: {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {e}")
            raise

        # 创建集合（使用版本化物理名称）
        self._ensure_collections()

    # ---------- 内部工具 ----------

    def _safe_model_tag(self, model_id: str) -> str:
        tag = (model_id or "").lower()
        # 替换非字母数字为短横线，压缩连续横线
        import re
        tag = re.sub(r"[^a-z0-9]+", "-", tag).strip("-")
        return tag

    def _embedder_id(self) -> str:
        provider = getattr(self.embedding_manager, "provider", "unknown")
        model_id = getattr(self.embedding_manager, "model_id", "model")
        dim = getattr(self.embedding_manager, "embedding_dim", "dim")
        return f"{provider}_{self._safe_model_tag(model_id)}_{dim}"

    def _physical_name(self, logical_name: str) -> str:
        return f"{logical_name}__{self._embedder_id()}"

    def _ensure_collections(self):
        """确保所有必要的集合存在"""
        try:
            # 物理名称带嵌入器后缀
            cars_name = self._physical_name("cars")
            analyses_name = self._physical_name("analyses")
            knowledge_name = self._physical_name("knowledge")
            market_name = self._physical_name("market_data")

            # 汽车数据集合
            self.cars_collection = self._get_or_create_collection(
                name=cars_name,
                metadata={
                    "description": "Car data embeddings",
                    "logical_name": "cars",
                    "embedder_id": self._embedder_id(),
                }
            )

            # 分析结果集合
            self.analyses_collection = self._get_or_create_collection(
                name=analyses_name,
                metadata={
                    "description": "Car analysis results embeddings",
                    "logical_name": "analyses",
                    "embedder_id": self._embedder_id(),
                }
            )

            # 知识库集合
            self.knowledge_collection = self._get_or_create_collection(
                name=knowledge_name,
                metadata={
                    "description": "Knowledge base embeddings",
                    "logical_name": "knowledge",
                    "embedder_id": self._embedder_id(),
                }
            )

            # 市场数据集合
            self.market_collection = self._get_or_create_collection(
                name=market_name,
                metadata={
                    "description": "Market data embeddings",
                    "logical_name": "market_data",
                    "embedder_id": self._embedder_id(),
                }
            )

            print("✅ All vector collections initialized")

        except Exception as e:
            logger.error(f"Error creating collections: {e}")
            raise

    def _get_or_create_collection(self, name: str, metadata: Dict[str, Any] = None):
        """获取或创建集合（按物理名）。不破坏已有不同维度的旧集合。"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(name=name)
            print(f"📚 Found existing collection: {name}")
            return collection
        except:
            # 创建新集合
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            print(f"🆕 Created new collection: {name}")
            return collection

    # =============== 汽车数据操作 ===============

    def add_car(self, car_id: int, car_data: Dict[str, Any]) -> bool:
        """添加汽车数据到向量存储

        Args:
            car_id: 汽车ID
            car_data: 汽车数据

        Returns:
            是否成功
        """
        try:
            # 创建描述性文本
            description = self.embedding_manager.create_car_description(car_data)

            # 生成嵌入
            embedding = self.embedding_manager.embed_text(description)

            # 准备元数据
            metadata = {
                "car_id": car_id,
                "make": car_data.get('make', ''),
                "model": car_data.get('model', ''),
                "year": car_data.get('year', 0),
                "price_paid": car_data.get('price_paid', 0.0),
                "mileage": car_data.get('mileage', 0),
                "type": "car_data"
            }

            # 添加到集合
            self.cars_collection.add(
                ids=[f"car_{car_id}"],
                embeddings=[embedding],
                documents=[description],
                metadatas=[metadata]
            )

            print(f"✅ Added car to vector store: {car_data.get('year')} {car_data.get('make')} {car_data.get('model')}")
            return True

        except Exception as e:
            logger.error(f"Error adding car to vector store: {e}")
            return False

    def search_similar_cars(self,
                           query_car: Dict[str, Any],
                           limit: int = 10,
                           similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """搜索相似汽车

        Args:
            query_car: 查询汽车数据
            limit: 结果限制
            similarity_threshold: 相似度阈值

        Returns:
            相似汽车列表
        """
        try:
            # 创建查询文本
            query_text = self.embedding_manager.create_car_description(query_car)

            # 生成查询嵌入
            query_embedding = self.embedding_manager.embed_text(query_text)

            # 搜索相似项
            results = self.cars_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )

            # 处理结果
            similar_cars = []
            if results['ids']:
                for i, car_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance  # 转换为相似度

                    if similarity >= similarity_threshold:
                        similar_cars.append({
                            'id': car_id,
                            'metadata': results['metadatas'][0][i],
                            'document': results['documents'][0][i],
                            'similarity': similarity
                        })

            return similar_cars

        except Exception as e:
            logger.error(f"Error searching similar cars: {e}")
            return []

    # =============== 分析结果操作 ===============

    def add_analysis(self, analysis_id: int, car_id: int, analysis_data: Dict[str, Any]) -> bool:
        """添加分析结果到向量存储

        Args:
            analysis_id: 分析ID
            car_id: 汽车ID
            analysis_data: 分析数据

        Returns:
            是否成功
        """
        try:
            # 创建描述性文本
            description = self.embedding_manager.create_analysis_description(analysis_data)

            # 生成嵌入
            embedding = self.embedding_manager.embed_text(description)

            # 准备元数据
            metadata = {
                "analysis_id": analysis_id,
                "car_id": car_id,
                "rule_based_score": analysis_data.get('rule_based_score', 0),
                "llm_score": analysis_data.get('llm_score', 0),
                "market_median_price": analysis_data.get('market_median_price', 0.0),
                "deal_category": analysis_data.get('deal_category', ''),
                "success": analysis_data.get('success', False),
                "type": "analysis_result"
            }

            # 添加到集合
            self.analyses_collection.add(
                ids=[f"analysis_{analysis_id}"],
                embeddings=[embedding],
                documents=[description],
                metadatas=[metadata]
            )

            print(f"✅ Added analysis to vector store: Analysis ID {analysis_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding analysis to vector store: {e}")
            return False

    def search_similar_analyses(self,
                              query_text: str,
                              limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似分析结果

        Args:
            query_text: 查询文本
            limit: 结果限制

        Returns:
            相似分析列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_manager.embed_text(query_text)

            # 搜索相似项
            results = self.analyses_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )

            # 处理结果
            similar_analyses = []
            if results['ids']:
                for i, analysis_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance

                    similar_analyses.append({
                        'id': analysis_id,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'similarity': similarity
                    })

            return similar_analyses

        except Exception as e:
            logger.error(f"Error searching similar analyses: {e}")
            return []

    # =============== 知识库操作 ===============

    def add_knowledge(self, knowledge_id: int, knowledge_data: Dict[str, Any]) -> bool:
        """添加知识库条目到向量存储

        Args:
            knowledge_id: 知识ID
            knowledge_data: 知识数据

        Returns:
            是否成功
        """
        try:
            # 创建完整文本
            text = self.embedding_manager.create_knowledge_text(knowledge_data)

            # 生成嵌入
            embedding = self.embedding_manager.embed_text(text)

            # 准备元数据
            metadata = {
                "knowledge_id": knowledge_id,
                "title": knowledge_data.get('title', ''),
                "content_type": knowledge_data.get('content_type', ''),
                "category": knowledge_data.get('category', ''),
                "source": knowledge_data.get('source', ''),
                "reliability_score": knowledge_data.get('reliability_score', 1.0),
                "type": "knowledge"
            }

            # 添加到集合
            self.knowledge_collection.add(
                ids=[f"knowledge_{knowledge_id}"],
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata]
            )

            print(f"✅ Added knowledge to vector store: {knowledge_data.get('title')}")
            return True

        except Exception as e:
            logger.error(f"Error adding knowledge to vector store: {e}")
            return False

    def search_knowledge(self,
                        query_text: str,
                        content_type: str = None,
                        category: str = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库

        Args:
            query_text: 查询文本
            content_type: 内容类型筛选
            category: 分类筛选
            limit: 结果限制

        Returns:
            知识条目列表
        """
        try:
            # 生成查询嵌入
            query_embedding = self.embedding_manager.embed_text(query_text)

            # 构建where条件
            where_conditions = {}
            if content_type:
                where_conditions["content_type"] = content_type
            if category:
                where_conditions["category"] = category

            # 搜索相似项
            results = self.knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_conditions if where_conditions else None,
                include=["documents", "metadatas", "distances"]
            )

            # 处理结果
            knowledge_items = []
            if results['ids']:
                for i, knowledge_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1 - distance

                    knowledge_items.append({
                        'id': knowledge_id,
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i],
                        'similarity': similarity
                    })

            return knowledge_items

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return []

    # =============== 通用搜索 ===============

    def semantic_search(self,
                       query_text: str,
                       collections: List[str] = None,
                       limit: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """跨集合语义搜索

        Args:
            query_text: 查询文本
            collections: 要搜索的集合列表
            limit: 每个集合的结果限制

        Returns:
            按集合分组的搜索结果
        """
        if collections is None:
            collections = ["cars", "analyses", "knowledge"]

        results = {}

        # 生成查询嵌入
        query_embedding = self.embedding_manager.embed_text(query_text)

        for collection_name in collections:
            try:
                collection = getattr(self, f"{collection_name}_collection")

                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    include=["documents", "metadatas", "distances"]
                )

                # 处理结果
                items = []
                if search_results['ids']:
                    for i, item_id in enumerate(search_results['ids'][0]):
                        distance = search_results['distances'][0][i]
                        similarity = 1 - distance

                        items.append({
                            'id': item_id,
                            'metadata': search_results['metadatas'][0][i],
                            'document': search_results['documents'][0][i],
                            'similarity': similarity,
                            'collection': collection_name
                        })

                results[collection_name] = items

            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
                results[collection_name] = []

        return results

    # =============== 统计和管理 ===============

    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = {}

            for collection_name in ["cars", "analyses", "knowledge", "market_data"]:
                collection = getattr(self, f"{collection_name}_collection")
                count = collection.count()
                stats[collection_name] = count

            stats['total_items'] = sum(stats.values())
            stats['embedding_info'] = self.embedding_manager.get_embedding_info()

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

    def delete_item(self, collection_name: str, item_id: str) -> bool:
        """删除向量存储中的项目

        Args:
            collection_name: 集合名称
            item_id: 项目ID

        Returns:
            是否成功
        """
        try:
            collection = getattr(self, f"{collection_name}_collection")
            collection.delete(ids=[item_id])
            print(f"✅ Deleted item {item_id} from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting item: {e}")
            return False

    def clear_collection(self, collection_name: str) -> bool:
        """清空集合

        Args:
            collection_name: 集合名称

        Returns:
            是否成功
        """
        try:
            # 删除现有集合
            self.client.delete_collection(name=collection_name)

            # 重新创建空集合
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"{collection_name} embeddings"}
            )

            # 更新实例变量
            setattr(self, f"{collection_name}_collection", collection)

            print(f"✅ Cleared collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
