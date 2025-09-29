#!/usr/bin/env python3
"""RAG系统初始化脚本 - 添加基础知识库"""

import sys
import os
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from database.manager import DatabaseManager
from rag.rag_system import RAGSystem


def setup_basic_knowledge(rag_system=None):
    """设置基础汽车知识库"""
    print("📚 Setting up basic car knowledge base...")

    try:
        if rag_system is None:
            rag_system = RAGSystem()
        db_manager = rag_system.db_manager

        # 基础汽车知识
        knowledge_entries = [
            {
                "title": "Toyota Camry可靠性指南",
                "content": """Toyota Camry是中型轿车市场的可靠选择。历年可靠性评分高，维修成本相对较低。
                2020年款Camry配备2.5L四缸发动机或3.5L V6发动机，油耗表现良好。
                二手车价值保持稳定，是家庭用车的热门选择。
                常见问题包括CVT变速箱在高里程时可能需要维护，但总体故障率低。""",
                "content_type": "vehicle_guide",
                "category": "reliability",
                "tags": ["Toyota", "Camry", "reliability", "maintenance"],
                "source": "automotive_knowledge_base"
            },
            {
                "title": "二手车价格评估要素",
                "content": """二手车价格主要受以下因素影响：
                1. 车龄和里程数 - 每年贬值率约10-20%，高里程影响更大
                2. 品牌和型号 - 可靠品牌如Toyota、Honda保值性更好
                3. 车况 - 事故记录、维修历史直接影响价值
                4. 市场需求 - 热门车型价格更稳定
                5. 地区差异 - 不同地区价格可能差异10-15%
                建议参考KBB、Edmunds等权威价格指南。""",
                "content_type": "pricing_guide",
                "category": "valuation",
                "tags": ["pricing", "valuation", "factors", "depreciation"],
                "source": "pricing_guide"
            },
            {
                "title": "购买二手车检查清单",
                "content": """购买二手车时必须检查的项目：
                外观：车身划痕、凹陷、锈蚀、油漆色差
                内饰：座椅磨损、电子设备功能、空调系统
                机械：发动机声音、变速箱换挡、刹车系统、轮胎磨损
                文件：车辆历史报告、维修记录、保险记录
                试驾：各种路况下的驾驶表现、异响、振动
                建议找专业技师进行全面检查，费用约100-200美元但能避免大损失。""",
                "content_type": "buying_guide",
                "category": "inspection",
                "tags": ["inspection", "checklist", "buying", "used_car"],
                "source": "buyer_guide"
            },
            {
                "title": "汽车里程数对价值的影响",
                "content": """里程数是影响二手车价值的关键因素：
                - 低里程（< 12,000英里/年）：溢价5-15%
                - 正常里程（12,000-15,000英里/年）：市场价格
                - 高里程（> 15,000英里/年）：贬值10-25%
                - 超高里程（> 100,000英里）：大幅贬值，需重点检查

                里程数vs车龄的权衡：
                - 3年6万英里 vs 5年4万英里，后者通常更值得购买
                - 高速公路里程比城市里程对车辆损耗更小
                重要的是维修记录比单纯里程数更能反映车况。""",
                "content_type": "technical_guide",
                "category": "mileage",
                "tags": ["mileage", "depreciation", "value", "wear"],
                "source": "technical_analysis"
            },
            {
                "title": "热门中型轿车品牌比较",
                "content": """中型轿车市场主要竞争者比较：
                Toyota Camry：可靠性高，保值性好，维修成本低，燃油经济性佳
                Honda Accord：驾驶体验优秀，空间大，可靠性高，技术配置丰富
                Nissan Altima：价格较低，舒适性好，但可靠性一般，CVT变速箱问题较多
                Hyundai Sonata：性价比高，保修期长，但保值性相对较差
                Mazda6：驾驶乐趣高，外观精美，但后排空间较小，保值性中等
                购买建议：Toyota和Honda是最安全的选择，Hyundai性价比最高。""",
                "content_type": "comparison_guide",
                "category": "brand_comparison",
                "tags": ["comparison", "midsize", "sedan", "brands"],
                "source": "market_analysis"
            }
        ]

        # 添加知识条目
        added_count = 0
        for knowledge in knowledge_entries:
            try:
                knowledge_id = db_manager.add_knowledge(**knowledge)

                # 同步到向量存储
                success = rag_system.vector_manager.add_knowledge(knowledge_id, knowledge)

                if success:
                    print(f"✅ Added: {knowledge['title']}")
                    added_count += 1
                else:
                    print(f"⚠️ DB added but vector sync failed: {knowledge['title']}")

            except Exception as e:
                print(f"❌ Failed to add: {knowledge['title']} - {e}")

        print(f"\n📚 Knowledge base setup completed: {added_count}/{len(knowledge_entries)} entries added")
        return added_count

    except Exception as e:
        print(f"❌ Error setting up knowledge base: {e}")
        return 0


def show_system_info():
    """显示系统信息"""
    print("\n🔧 System Information")
    print("=" * 50)

    try:
        # 数据库统计
        db_manager = DatabaseManager()
        db_stats = db_manager.get_stats()

        print("Database:")
        for key, value in db_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            else:
                print(f"  {key}: {value}")

        # RAG系统统计
        rag_system = RAGSystem()
        rag_stats = rag_system.get_system_stats()

        vector_stats = rag_stats.get('vector_store_stats', {})
        print("\nVector Store:")
        for collection, count in vector_stats.items():
            if collection != 'embedding_info':
                print(f"  {collection}: {count:,} items")

        embedding_info = rag_stats.get('embedding_info', {})
        print("\nEmbedding Model:")
        for key, value in embedding_info.items():
            print(f"  {key}: {value}")

    except Exception as e:
        print(f"❌ Error getting system info: {e}")


def main():
    """主函数"""
    print("🚀 RAG-Enhanced Car Analysis System Setup")
    print("=" * 60)

    # 检查环境
    print("Checking environment...")

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set. Using HuggingFace embeddings instead.")
    else:
        print("✅ OpenAI API key found")

    # 初始化系统
    print("\nInitializing RAG system...")
    try:
        # 先尝试完整初始化
        rag_system = RAGSystem()
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"⚠️ RAG system initialization failed: {e}")
        print("   Trying database-only mode...")
        try:
            # 仅初始化数据库和向量存储
            from database.manager import DatabaseManager
            from rag.vector_store import VectorStoreManager
            from rag.embeddings import EmbeddingManager

            db_manager = DatabaseManager()
            embedding_manager = EmbeddingManager()
            vector_manager = VectorStoreManager(embedding_manager=embedding_manager)

            print("✅ Database and vector store initialized successfully")
            print("⚠️ Note: LLM features will not be available without OpenAI API key")

            # 创建一个简化的RAG系统类来保存知识
            class SimpleRAGSystem:
                def __init__(self, db_manager, vector_manager):
                    self.db_manager = db_manager
                    self.vector_manager = vector_manager

            rag_system = SimpleRAGSystem(db_manager, vector_manager)

        except Exception as e2:
            print(f"❌ Complete initialization failed: {e2}")
            return

    # 设置知识库
    added_count = setup_basic_knowledge(rag_system)

    # 显示系统信息
    show_system_info()

    # 完成提示
    print("\n" + "=" * 60)
    print("🎉 Setup Complete!")
    print("=" * 60)

    if added_count > 0:
        print(f"✅ Added {added_count} knowledge base entries")
        print("\nNext steps:")
        print("1. Run tests: python test_rag_system.py")
        print("2. Use CLI tool: python db_manager_cli.py")
        print("3. Analyze cars with enhanced RAG capabilities")
    else:
        print("⚠️ No knowledge entries were added. Check the error messages above.")

    print("\n💡 To add more knowledge, use the CLI tool's interactive mode.")


if __name__ == "__main__":
    main()