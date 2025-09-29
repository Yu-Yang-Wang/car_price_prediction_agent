#!/usr/bin/env python3
"""简单的数据库功能测试脚本"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from database.manager import DatabaseManager


def test_database_functionality():
    """测试数据库基本功能"""
    print("🧪 Testing Database Functionality")
    print("=" * 50)

    try:
        # 初始化数据库管理器
        db_manager = DatabaseManager()
        print("✅ Database manager initialized")

        # 测试统计功能
        stats = db_manager.get_stats()
        print("\n📊 Database Statistics:")
        for key, value in stats.items():
            if key == 'success_rate':
                print(f"   {key}: {value:.1f}%")
            elif key == 'last_analysis':
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value:,}")

        # 测试知识库搜索
        print("\n🔍 Testing Knowledge Base Search:")
        knowledge_results = db_manager.search_knowledge("Toyota Camry", limit=3)
        if knowledge_results:
            print(f"   Found {len(knowledge_results)} knowledge entries:")
            for entry in knowledge_results:
                print(f"   - [{entry['id']}] {entry['title']}")
                print(f"     Type: {entry['content_type']} | Category: {entry.get('category', 'N/A')}")
        else:
            print("   No knowledge entries found")

        # 测试热门品牌
        print("\n🏆 Testing Popular Makes:")
        popular_makes = db_manager.get_popular_makes(limit=5)
        if popular_makes:
            for i, make_info in enumerate(popular_makes, 1):
                make = make_info['make']
                count = make_info['count']
                avg_price = make_info['avg_price']
                print(f"   {i}. {make}: {count} cars | Avg: ${avg_price:,.0f}")
        else:
            print("   No car data available yet")

        print("\n✅ Database functionality test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_database():
    """测试向量数据库功能"""
    print("\n🧪 Testing Vector Database Functionality")
    print("=" * 50)

    try:
        from rag.embeddings import EmbeddingManager
        from rag.vector_store import VectorStoreManager

        # 初始化组件
        embedding_manager = EmbeddingManager()
        vector_manager = VectorStoreManager(embedding_manager=embedding_manager)
        print("✅ Vector store components initialized")

        # 测试集合统计
        stats = vector_manager.get_collection_stats()
        print("\n📊 Vector Store Statistics:")
        for collection, count in stats.items():
            if collection != 'embedding_info':
                print(f"   {collection}: {count:,} items")

        # 测试知识库搜索
        print("\n🔍 Testing Vector Knowledge Search:")
        knowledge_results = vector_manager.search_knowledge("Toyota reliability", limit=3)
        if knowledge_results:
            print(f"   Found {len(knowledge_results)} similar knowledge entries:")
            for result in knowledge_results:
                metadata = result.get('metadata', {})
                similarity = result.get('similarity', 0)
                print(f"   - {metadata.get('title', 'N/A')} (similarity: {similarity:.3f})")
        else:
            print("   No similar knowledge found")

        print("\n✅ Vector database test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Vector database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 RAG-Enhanced Car Analysis System Test")
    print("=" * 80)

    # 测试数据库
    db_success = test_database_functionality()

    # 测试向量数据库
    vector_success = test_vector_database()

    # 总结
    print("\n" + "=" * 80)
    print("🏁 Test Summary")
    print("=" * 80)

    total_tests = 2
    passed_tests = sum([db_success, vector_success])

    print(f"Tests passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("🎉 All tests passed! The RAG system database is working correctly.")
        print("\n💡 System ready for:")
        print("   - Car analysis with historical context")
        print("   - Knowledge-enhanced recommendations")
        print("   - Similarity-based case retrieval")
        print("   - Data persistence and analytics")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")

    print(f"\n📝 Note: LLM features require OPENAI_API_KEY for full functionality")