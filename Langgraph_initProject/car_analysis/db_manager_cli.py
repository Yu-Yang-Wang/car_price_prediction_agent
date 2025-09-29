#!/usr/bin/env python3
"""数据库和RAG系统管理工具 - 命令行界面"""

import sys
import os
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import json

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Make sure OPENAI_API_KEY is set as environment variable.")

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    # 加载 .env 中的环境变量（如 OPENAI_API_KEY, EMBEDDINGS_PROVIDER 等）
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from database.manager import DatabaseManager
from rag.rag_system import RAGSystem


class DatabaseManagerCLI:
    """数据库管理器命令行界面"""

    def __init__(self):
        self.db_manager = DatabaseManager()
        try:
            self.rag_system = RAGSystem(db_manager=self.db_manager)
            print("🚀 Database and RAG System Manager initialized")
        except Exception as e:
            print(f"⚠️ RAG System failed to initialize: {e}")
            print("🚀 Database Manager initialized (RAG features limited)")
            self.rag_system = None

    # =============== 统计和查看 ===============

    def show_stats(self):
        """显示数据库统计信息"""
        print("\n📊 Database Statistics")
        print("=" * 50)

        stats = self.db_manager.get_stats()
        for key, value in stats.items():
            if key == 'success_rate':
                print(f"   {key}: {value:.1f}%")
            elif key == 'last_analysis':
                print(f"   {key}: {value}")
            else:
                print(f"   {key}: {value:,}")

        if self.rag_system:
            print("\n🧠 RAG System Statistics")
            print("=" * 50)

            rag_stats = self.rag_system.get_system_stats()
            vector_stats = rag_stats.get('vector_store_stats', {})
            embedding_info = rag_stats.get('embedding_info', {})

            print("Vector Store:")
            for collection, count in vector_stats.items():
                if collection != 'embedding_info':
                    print(f"   {collection}: {count:,} items")

            print("\nEmbedding Model:")
            for key, value in embedding_info.items():
                print(f"   {key}: {value}")
        else:
            print("\n⚠️ RAG System not available")

    def list_cars(self, limit: int = 20):
        """列出汽车数据"""
        print(f"\n🚗 Recent Cars (limit: {limit})")
        print("=" * 80)

        cars = self.db_manager.search_cars(limit=limit)
        if not cars:
            print("   No cars found in database")
            return

        for car in cars:
            analysis_info = ""
            car_with_analysis = self.db_manager.get_car_with_analysis(car['id'])
            if car_with_analysis and car_with_analysis.get('analysis'):
                analysis = car_with_analysis['analysis']
                score = analysis.get('rule_based_score', 'N/A')
                verdict = analysis.get('rule_based_verdict', 'N/A')
                analysis_info = f" | Score: {score}/100 ({verdict})"

            print(f"   [{car['id']}] {car['year']} {car['make']} {car['model']} - "
                  f"${car['price_paid']:,.0f} | {car['mileage']:,} mi{analysis_info}")

    def search_cars_interactive(self):
        """交互式搜索汽车"""
        print("\n🔍 Interactive Car Search")
        print("=" * 40)

        make = input("Brand (optional): ").strip() or None
        model = input("Model (optional): ").strip() or None

        year_input = input("Year range (e.g., 2018-2022, optional): ").strip()
        year_range = None
        if year_input and '-' in year_input:
            try:
                start, end = year_input.split('-')
                year_range = (int(start), int(end))
            except ValueError:
                print("Invalid year range format")

        price_input = input("Price range (e.g., 20000-30000, optional): ").strip()
        price_range = None
        if price_input and '-' in price_input:
            try:
                start, end = price_input.split('-')
                price_range = (float(start), float(end))
            except ValueError:
                print("Invalid price range format")

        limit = 20
        try:
            limit_input = input(f"Limit (default {limit}): ").strip()
            if limit_input:
                limit = int(limit_input)
        except ValueError:
            pass

        print("\nSearching...")
        cars = self.db_manager.search_cars(
            make=make,
            model=model,
            year_range=year_range,
            price_range=price_range,
            limit=limit
        )

        if not cars:
            print("No cars found matching your criteria")
            return

        print(f"\nFound {len(cars)} cars:")
        for car in cars:
            print(f"   [{car['id']}] {car['year']} {car['make']} {car['model']} - "
                  f"${car['price_paid']:,.0f} | {car['mileage']:,} mi")

    def show_popular_makes(self):
        """显示热门品牌"""
        print("\n🏆 Popular Car Makes")
        print("=" * 50)

        popular_makes = self.db_manager.get_popular_makes(limit=10)
        if not popular_makes:
            print("   No data available")
            return

        for i, make_info in enumerate(popular_makes, 1):
            make = make_info['make']
            count = make_info['count']
            avg_price = make_info['avg_price']
            print(f"   {i:2d}. {make:<15} {count:3d} cars | Avg: ${avg_price:,.0f}")

    # =============== 知识库管理 ===============

    def add_knowledge_interactive(self):
        """交互式添加知识库条目"""
        print("\n📚 Add Knowledge Base Entry")
        print("=" * 40)

        title = input("Title: ").strip()
        if not title:
            print("Title is required")
            return

        print("Content (press Enter twice to finish):")
        content_lines = []
        while True:
            line = input()
            if line == "" and content_lines and content_lines[-1] == "":
                break
            content_lines.append(line)

        content = "\n".join(content_lines).strip()
        if not content:
            print("Content is required")
            return

        content_type = input("Content type (default: general): ").strip() or "general"
        category = input("Category (optional): ").strip() or None
        source = input("Source (optional): ").strip() or None

        tags_input = input("Tags (comma-separated, optional): ").strip()
        tags = [tag.strip() for tag in tags_input.split(',')] if tags_input else None

        try:
            knowledge_id = self.db_manager.add_knowledge(
                title=title,
                content=content,
                content_type=content_type,
                category=category,
                tags=tags,
                source=source
            )

            # 同步到向量存储
            knowledge_data = {
                'title': title,
                'content': content,
                'content_type': content_type,
                'category': category,
                'tags': tags,
                'source': source,
                'reliability_score': 1.0
            }

            success = self.rag_system.vector_manager.add_knowledge(knowledge_id, knowledge_data)
            if success:
                print(f"✅ Knowledge added successfully (ID: {knowledge_id})")
            else:
                print(f"⚠️ Knowledge added to DB but failed to sync to vector store")

        except Exception as e:
            print(f"❌ Error adding knowledge: {e}")

    def search_knowledge_interactive(self):
        """交互式搜索知识库"""
        print("\n🔍 Search Knowledge Base")
        print("=" * 40)

        query = input("Search query: ").strip()
        if not query:
            print("Query is required")
            return

        content_type = input("Content type filter (optional): ").strip() or None
        category = input("Category filter (optional): ").strip() or None

        try:
            limit = int(input("Limit (default 10): ").strip() or "10")
        except ValueError:
            limit = 10

        results = self.db_manager.search_knowledge(
            query=query,
            content_type=content_type,
            category=category,
            limit=limit
        )

        if not results:
            print("No knowledge entries found")
            return

        print(f"\nFound {len(results)} knowledge entries:")
        for entry in results:
            print(f"\n   [{entry['id']}] {entry['title']}")
            print(f"   Type: {entry['content_type']} | Category: {entry.get('category', 'N/A')}")
            print(f"   Content: {entry['content'][:100]}...")
            if entry.get('tags'):
                print(f"   Tags: {', '.join(entry['tags'])}")

    # =============== RAG系统测试 ===============

    async def test_rag_system(self):
        """测试RAG系统"""
        print("\n🧠 Testing RAG System")
        print("=" * 40)

        # 测试问答
        test_questions = [
            "什么是好的汽车交易？",
            "如何评估二手车价格？",
            "Toyota Camry的可靠性如何？",
            "购买二手车需要注意什么？"
        ]

        for question in test_questions:
            print(f"\n❓ Question: {question}")
            print("-" * 50)

            try:
                result = self.rag_system.answer_question(question)
                answer = result.get('answer', 'No answer generated')
                confidence = result.get('confidence', 0)

                print(f"🤖 Answer (confidence: {confidence:.2f}):")
                print(f"   {answer[:200]}...")

            except Exception as e:
                print(f"❌ Error: {e}")

    async def rag_qa_interactive(self):
        """交互式RAG问答"""
        print("\n🤖 Interactive RAG Q&A")
        print("=" * 40)
        print("Type 'quit' to exit")

        while True:
            question = input("\n❓ Your question: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            try:
                result = self.rag_system.answer_question(question)
                answer = result.get('answer', 'No answer generated')
                confidence = result.get('confidence', 0)

                print(f"\n🤖 Answer (confidence: {confidence:.2f}):")
                print(f"{answer}")

                # 显示检索信息（可选）
                show_context = input("\nShow retrieved context? (y/N): ").strip().lower()
                if show_context == 'y':
                    retrieved_info = result.get('retrieved_info', 'No context')
                    print(f"\n📚 Retrieved Context:")
                    print(f"{retrieved_info[:500]}...")

            except Exception as e:
                print(f"❌ Error: {e}")

    # =============== 数据导入导出 ===============

    def export_data(self, filename: str = None):
        """导出数据库数据"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"outputs/car_analysis_export_{timestamp}.json"

        print(f"\n📤 Exporting data to {filename}")

        try:
            # 导出汽车数据
            cars = self.db_manager.search_cars(limit=1000)

            # 为每辆车添加分析数据
            for car in cars:
                car_with_analysis = self.db_manager.get_car_with_analysis(car['id'])
                if car_with_analysis and car_with_analysis.get('analysis'):
                    car['analysis'] = car_with_analysis['analysis']

            # 导出知识库
            knowledge = self.db_manager.search_knowledge(limit=1000)

            # 导出统计信息
            stats = self.db_manager.get_stats()

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "stats": stats,
                "cars": cars,
                "knowledge": knowledge
            }

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            print(f"✅ Data exported successfully")
            print(f"   Cars: {len(cars)}")
            print(f"   Knowledge entries: {len(knowledge)}")

        except Exception as e:
            print(f"❌ Export failed: {e}")

    def sync_to_vector_store(self):
        """同步数据到向量存储"""
        print("\n🔄 Syncing data to vector store")
        print("=" * 40)

        try:
            # 同步汽车数据
            cars = self.db_manager.search_cars(limit=1000)
            car_success_count = 0

            print(f"Syncing {len(cars)} cars...")
            for car in cars:
                if self.rag_system.sync_car_to_vector_store(car['id']):
                    car_success_count += 1

            # 同步分析数据
            analysis_success_count = 0
            print("Syncing analysis data...")
            for car in cars:
                car_with_analysis = self.db_manager.get_car_with_analysis(car['id'])
                if car_with_analysis and car_with_analysis.get('analysis'):
                    analysis_id = car_with_analysis['analysis'].get('id')
                    if analysis_id and self.rag_system.sync_analysis_to_vector_store(analysis_id, car['id']):
                        analysis_success_count += 1

            # 同步知识库
            knowledge_entries = self.db_manager.search_knowledge(limit=1000)
            knowledge_success_count = 0

            print(f"Syncing {len(knowledge_entries)} knowledge entries...")
            for entry in knowledge_entries:
                if self.rag_system.vector_manager.add_knowledge(entry['id'], entry):
                    knowledge_success_count += 1

            print(f"\n✅ Sync completed:")
            print(f"   Cars: {car_success_count}/{len(cars)}")
            print(f"   Analyses: {analysis_success_count}")
            print(f"   Knowledge: {knowledge_success_count}/{len(knowledge_entries)}")

        except Exception as e:
            print(f"❌ Sync failed: {e}")

    # =============== 主菜单 ===============

    def show_menu(self):
        """显示主菜单"""
        print("\n" + "=" * 60)
        print("🗄️  Car Analysis Database & RAG Manager")
        print("=" * 60)
        print("1.  📊 Show Statistics")
        print("2.  🚗 List Recent Cars")
        print("3.  🔍 Search Cars")
        print("4.  🏆 Show Popular Makes")
        print("5.  📚 Add Knowledge Entry")
        print("6.  🔍 Search Knowledge")
        print("7.  🧠 Test RAG System")
        print("8.  🤖 Interactive Q&A")
        print("9.  📤 Export Data")
        print("10. 🔄 Sync to Vector Store")
        print("0.  ❌ Exit")
        print("=" * 60)

    async def run_interactive(self):
        """运行交互式界面"""
        while True:
            self.show_menu()
            choice = input("\nSelect option: ").strip()

            try:
                if choice == '1':
                    self.show_stats()
                elif choice == '2':
                    try:
                        limit = int(input("Limit (default 20): ").strip() or "20")
                    except ValueError:
                        limit = 20
                    self.list_cars(limit)
                elif choice == '3':
                    self.search_cars_interactive()
                elif choice == '4':
                    self.show_popular_makes()
                elif choice == '5':
                    self.add_knowledge_interactive()
                elif choice == '6':
                    self.search_knowledge_interactive()
                elif choice == '7':
                    await self.test_rag_system()
                elif choice == '8':
                    await self.rag_qa_interactive()
                elif choice == '9':
                    filename = input("Export filename (optional): ").strip() or None
                    self.export_data(filename)
                elif choice == '10':
                    self.sync_to_vector_store()
                elif choice == '0':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option")

            except KeyboardInterrupt:
                print("\n\n⏹️ Operation interrupted")
                continue
            except Exception as e:
                print(f"❌ Error: {e}")

            input("\nPress Enter to continue...")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Database and RAG System Manager')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--export', type=str, help='Export data to file')
    parser.add_argument('--sync', action='store_true', help='Sync data to vector store')

    args = parser.parse_args()

    try:
        manager = DatabaseManagerCLI()

        if args.stats:
            manager.show_stats()
        elif args.export:
            manager.export_data(args.export)
        elif args.sync:
            manager.sync_to_vector_store()
        else:
            # 运行交互式界面
            asyncio.run(manager.run_interactive())

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
