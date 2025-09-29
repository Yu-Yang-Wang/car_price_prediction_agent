#!/usr/bin/env python3
"""测试完整的RAG增强汽车分析系统"""

import sys
import os
import asyncio
from datetime import datetime

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.models import CarAnalysisState
from core.graph import build_single_car_graph
from database.manager import DatabaseManager
from rag.rag_system import RAGSystem


def create_test_car_data():
    """创建测试用汽车数据"""
    return {
        "year": 2020,
        "make": "Toyota",
        "model": "Camry",
        "trim": "LE",
        "mileage": 35000,
        "price_paid": 22000,
        "color": "White",
        "transmission": "Automatic",
        "fuel_type": "Gasoline",
        "condition": "Good",
        "location": "California",
        "session_id": f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }


async def test_single_car_analysis():
    """测试单个汽车的RAG增强分析"""
    print("🚗 Testing Single Car RAG-Enhanced Analysis")
    print("=" * 60)

    # 创建测试数据
    test_car = create_test_car_data()

    # 初始化状态
    initial_state = CarAnalysisState(
        current_car=test_car,
        session_id=test_car["session_id"]
    )

    print(f"Testing car: {test_car['year']} {test_car['make']} {test_car['model']}")
    print(f"Price: ${test_car['price_paid']:,} | Mileage: {test_car['mileage']:,} mi")
    print()

    try:
        # 构建并运行工作流
        graph = build_single_car_graph()

        print("🔄 Running RAG-enhanced analysis workflow...")
        result = await graph.ainvoke(initial_state)

        # 显示结果
        print("\n📊 Analysis Results:")
        print("-" * 40)

        # 价格研究结果
        price_research = result.get("price_research", {})
        if price_research.get("success"):
            print(f"✅ Price Research: {price_research.get('comparable_count', 0)} comparables found")
            if price_research.get("rag_enhanced"):
                print(f"   🧠 RAG Enhanced: {price_research.get('rag_similarity_count', 0)} similar cases")

        # 价格比较结果
        price_comparison = result.get("price_comparison", {})
        if price_comparison.get("success"):
            market_price = price_comparison.get("market_median_price", 0)
            price_delta = price_comparison.get("price_delta", 0)
            print(f"💰 Market Analysis: ${market_price:,.0f} median | ${price_delta:+,.0f} difference")

        # 规则评分结果
        deal_score = result.get("deal_score", {})
        if deal_score.get("success"):
            score = deal_score.get("score", 0)
            verdict = deal_score.get("verdict", "Unknown")
            print(f"📐 Rule-based Score: {score}/100 ({verdict})")

            # RAG调整评分
            if deal_score.get("rag_adjusted_score"):
                rag_score = deal_score.get("rag_adjusted_score")
                print(f"   🧠 RAG Adjusted: {rag_score}/100 (based on {deal_score.get('rag_similar_cases_count', 0)} cases)")

        # LLM意见结果
        llm_opinion = result.get("llm_opinion", {})
        if llm_opinion.get("success"):
            llm_score = llm_opinion.get("score", 0)
            llm_verdict = llm_opinion.get("verdict", "Unknown")
            print(f"🤖 LLM Opinion: {llm_score}/100 ({llm_verdict})")

            if llm_opinion.get("rag_enhanced"):
                confidence = llm_opinion.get("rag_confidence", 0)
                print(f"   🧠 RAG Enhanced: Confidence {confidence:.2f}")
                print(f"   💭 Reasoning: {llm_opinion.get('reasoning', '')[:100]}...")

        # 数据库保存结果
        if result.get("database_saved"):
            car_id = result.get("database_car_id")
            analysis_id = result.get("database_analysis_id")
            print(f"💾 Database: Saved (Car ID: {car_id}, Analysis ID: {analysis_id})")

        print("\n✅ Single car analysis test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_rag_qa_system():
    """测试RAG问答系统"""
    print("\n🤖 Testing RAG Q&A System")
    print("=" * 60)

    try:
        rag_system = RAGSystem()

        # 测试问题
        test_questions = [
            "What should I look for when buying a used Toyota Camry?",
            "Is $22,000 a good price for a 2020 Toyota Camry with 35,000 miles?",
            "How does mileage affect car value?"
        ]

        for i, question in enumerate(test_questions, 1):
            print(f"\n❓ Question {i}: {question}")
            print("-" * 50)

            result = rag_system.answer_question(question)

            answer = result.get('answer', 'No answer generated')
            confidence = result.get('confidence', 0)

            print(f"🤖 Answer (confidence: {confidence:.2f}):")
            print(f"   {answer[:200]}{'...' if len(answer) > 200 else ''}")

        print("\n✅ RAG Q&A system test completed!")
        return True

    except Exception as e:
        print(f"\n❌ RAG Q&A test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database_connection():
    """测试数据库连接"""
    print("\n📊 Testing Database Connection")
    print("=" * 60)

    try:
        db_manager = DatabaseManager()
        stats = db_manager.get_stats()

        print("Database Statistics:")
        for key, value in stats.items():
            if key == 'success_rate':
                print(f"  {key}: {value:.1f}%")
            elif key == 'last_analysis':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:,}")

        print("\n✅ Database connection test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Database connection test failed: {e}")
        return False


async def main():
    """主测试函数"""
    print("🧪 RAG-Enhanced Car Analysis System Test Suite")
    print("=" * 80)

    test_results = []

    # 测试数据库连接
    print("\n1. Database Connection Test")
    test_results.append(test_database_connection())

    # 测试RAG问答系统
    print("\n2. RAG Q&A System Test")
    test_results.append(await test_rag_qa_system())

    # 测试完整汽车分析流程
    print("\n3. Complete Car Analysis Test")
    test_results.append(await test_single_car_analysis())

    # 显示测试总结
    print("\n" + "=" * 80)
    print("🏁 Test Summary")
    print("=" * 80)

    passed = sum(test_results)
    total = len(test_results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! RAG system is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")

    print("\n💡 Next steps:")
    print("- Use the CLI tool: python db_manager_cli.py")
    print("- Add knowledge to the system for better responses")
    print("- Test with real car data")


if __name__ == "__main__":
    asyncio.run(main())