"""Database manager for car analysis system"""

import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import (
    Base, Car, CarAnalysis, MarketData, AnalysisSession,
    KnowledgeBase, UserQuery, DatabaseHelper
)


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, db_path: str = "database/car_analysis.db"):
        """初始化数据库管理器

        Args:
            db_path: 数据库文件路径
        """
        # 确保数据库目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # 创建数据库引擎
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)

        # 创建会话工厂
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        # 创建所有表
        Base.metadata.create_all(self.engine)

        print(f"📊 Database initialized: {db_path}")

    def get_session(self) -> Session:
        """获取数据库会话"""
        return self.SessionLocal()

    # =============== 汽车数据操作 ===============

    def save_car(self, car_data: Dict[str, Any], session_id: Optional[str] = None) -> int:
        """保存汽车数据

        Args:
            car_data: 汽车数据字典
            session_id: 分析会话ID

        Returns:
            car_id: 汽车ID
        """
        with self.get_session() as session:
            try:
                car = Car(
                    make=car_data.get('make', ''),
                    model=car_data.get('model', ''),
                    year=car_data.get('year', 0),
                    mileage=car_data.get('mileage', 0),
                    price_paid=car_data.get('price_paid', 0.0),
                    trim=car_data.get('trim'),
                    color=car_data.get('color'),
                    transmission=car_data.get('transmission'),
                    engine=car_data.get('engine'),
                    fuel_type=car_data.get('fuel_type'),
                    condition=car_data.get('condition'),
                    location=car_data.get('location'),
                    pdf_source=car_data.get('pdf_source'),
                    pdf_page=car_data.get('pdf_page'),
                    raw_text=car_data.get('raw_text', '')
                )

                session.add(car)
                session.commit()
                session.refresh(car)

                print(f"✅ Saved car: {car.year} {car.make} {car.model} (ID: {car.id})")
                return car.id

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error saving car: {e}")
                raise

    def get_car(self, car_id: int) -> Optional[Dict[str, Any]]:
        """获取汽车数据"""
        with self.get_session() as session:
            car = session.query(Car).filter(Car.id == car_id).first()
            if car:
                return DatabaseHelper.car_to_dict(car)
            return None

    def search_cars(self,
                   make: str = None,
                   model: str = None,
                   year_range: tuple = None,
                   price_range: tuple = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        """搜索汽车数据

        Args:
            make: 品牌筛选
            model: 型号筛选
            year_range: 年份范围 (min_year, max_year)
            price_range: 价格范围 (min_price, max_price)
            limit: 结果限制

        Returns:
            汽车数据列表
        """
        with self.get_session() as session:
            query = session.query(Car)

            if make:
                query = query.filter(Car.make.ilike(f"%{make}%"))
            if model:
                query = query.filter(Car.model.ilike(f"%{model}%"))
            if year_range:
                min_year, max_year = year_range
                query = query.filter(Car.year >= min_year, Car.year <= max_year)
            if price_range:
                min_price, max_price = price_range
                query = query.filter(Car.price_paid >= min_price, Car.price_paid <= max_price)

            cars = query.order_by(desc(Car.created_at)).limit(limit).all()
            return [DatabaseHelper.car_to_dict(car) for car in cars]

    # =============== 分析结果操作 ===============

    def save_analysis(self, car_id: int, analysis_data: Dict[str, Any]) -> int:
        """保存分析结果

        Args:
            car_id: 汽车ID
            analysis_data: 分析数据

        Returns:
            analysis_id: 分析ID
        """
        with self.get_session() as session:
            try:
                analysis = CarAnalysis(
                    car_id=car_id,
                    rule_based_score=analysis_data.get('rule_based_score'),
                    rule_based_verdict=analysis_data.get('rule_based_verdict'),
                    llm_score=analysis_data.get('llm_score'),
                    llm_verdict=analysis_data.get('llm_verdict'),
                    llm_reasoning=analysis_data.get('llm_reasoning'),
                    market_median_price=analysis_data.get('market_median_price'),
                    price_delta=analysis_data.get('price_delta'),
                    price_delta_percent=analysis_data.get('price_delta_percent'),
                    deal_category=analysis_data.get('deal_category'),
                    data_source=analysis_data.get('data_source'),
                    comparable_count=analysis_data.get('comparable_count'),
                    research_quality=analysis_data.get('research_quality'),
                    success=analysis_data.get('success', False),
                    error_message=analysis_data.get('error_message'),
                    analysis_version=analysis_data.get('analysis_version', '1.0'),
                    full_analysis_data=analysis_data
                )

                session.add(analysis)
                session.commit()
                session.refresh(analysis)

                print(f"✅ Saved analysis for car ID {car_id} (Analysis ID: {analysis.id})")
                return analysis.id

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error saving analysis: {e}")
                raise

    def get_car_with_analysis(self, car_id: int) -> Optional[Dict[str, Any]]:
        """获取汽车及其分析数据"""
        with self.get_session() as session:
            car = session.query(Car).filter(Car.id == car_id).first()
            if not car:
                return None

            car_dict = DatabaseHelper.car_to_dict(car)

            # 获取最新的分析结果
            latest_analysis = session.query(CarAnalysis).filter(
                CarAnalysis.car_id == car_id
            ).order_by(desc(CarAnalysis.created_at)).first()

            if latest_analysis:
                car_dict['analysis'] = DatabaseHelper.analysis_to_dict(latest_analysis)

            return car_dict

    # =============== 市场数据操作 ===============

    def save_market_data(self, car_id: int, market_data_list: List[Dict[str, Any]]):
        """保存市场数据

        Args:
            car_id: 汽车ID
            market_data_list: 市场数据列表
        """
        with self.get_session() as session:
            try:
                for data in market_data_list:
                    market_data = MarketData(
                        car_id=car_id,
                        search_query=data.get('search_query'),
                        search_engine=data.get('search_engine', 'tavily'),
                        comparable_make=data.get('make'),
                        comparable_model=data.get('model'),
                        comparable_year=data.get('year'),
                        comparable_mileage=data.get('mileage'),
                        comparable_price=data.get('price'),
                        comparable_url=data.get('url'),
                        comparable_source=data.get('source'),
                        similarity_score=data.get('similarity_score', 1.0)
                    )
                    session.add(market_data)

                session.commit()
                print(f"✅ Saved {len(market_data_list)} market data entries for car ID {car_id}")

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error saving market data: {e}")
                raise

    # =============== 知识库操作 ===============

    def add_knowledge(self,
                     title: str,
                     content: str,
                     content_type: str = "general",
                     category: str = None,
                     tags: List[str] = None,
                     source: str = None) -> int:
        """添加知识库条目

        Args:
            title: 标题
            content: 内容
            content_type: 内容类型
            category: 分类
            tags: 标签列表
            source: 来源

        Returns:
            knowledge_id: 知识ID
        """
        with self.get_session() as session:
            try:
                knowledge = KnowledgeBase(
                    title=title,
                    content=content,
                    content_type=content_type,
                    category=category,
                    tags=tags or [],
                    source=source,
                    reliability_score=1.0
                )

                session.add(knowledge)
                session.commit()
                session.refresh(knowledge)

                print(f"✅ Added knowledge: {title} (ID: {knowledge.id})")
                return knowledge.id

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error adding knowledge: {e}")
                raise

    def search_knowledge(self,
                        query: str = None,
                        content_type: str = None,
                        category: str = None,
                        limit: int = 10) -> List[Dict[str, Any]]:
        """搜索知识库

        Args:
            query: 搜索查询
            content_type: 内容类型筛选
            category: 分类筛选
            limit: 结果限制

        Returns:
            知识条目列表
        """
        with self.get_session() as session:
            query_obj = session.query(KnowledgeBase)

            if query:
                query_obj = query_obj.filter(
                    KnowledgeBase.title.contains(query) |
                    KnowledgeBase.content.contains(query)
                )
            if content_type:
                query_obj = query_obj.filter(KnowledgeBase.content_type == content_type)
            if category:
                query_obj = query_obj.filter(KnowledgeBase.category == category)

            results = query_obj.order_by(desc(KnowledgeBase.reliability_score)).limit(limit).all()

            return [{
                'id': kb.id,
                'title': kb.title,
                'content': kb.content,
                'content_type': kb.content_type,
                'category': kb.category,
                'tags': kb.tags,
                'source': kb.source,
                'reliability_score': kb.reliability_score,
                'created_at': kb.created_at.isoformat() if kb.created_at else None
            } for kb in results]

    # =============== 分析会话管理 ===============

    def create_session(self, pdf_path: str = None) -> str:
        """创建分析会话

        Args:
            pdf_path: PDF文件路径

        Returns:
            session_id: 会话ID
        """
        session_id = str(uuid.uuid4())

        with self.get_session() as session:
            try:
                analysis_session = AnalysisSession(
                    session_id=session_id,
                    pdf_path=pdf_path,
                    started_at=datetime.utcnow()
                )

                session.add(analysis_session)
                session.commit()

                print(f"✅ Created analysis session: {session_id}")
                return session_id

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error creating session: {e}")
                raise

    def update_session(self, session_id: str, **kwargs):
        """更新分析会话"""
        with self.get_session() as session:
            try:
                analysis_session = session.query(AnalysisSession).filter(
                    AnalysisSession.session_id == session_id
                ).first()

                if analysis_session:
                    for key, value in kwargs.items():
                        if hasattr(analysis_session, key):
                            setattr(analysis_session, key, value)

                    session.commit()
                    print(f"✅ Updated session: {session_id}")

            except SQLAlchemyError as e:
                session.rollback()
                print(f"❌ Error updating session: {e}")
                raise

    # =============== 统计和分析 ===============

    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        with self.get_session() as session:
            stats = {
                'total_cars': session.query(Car).count(),
                'total_analyses': session.query(CarAnalysis).count(),
                'successful_analyses': session.query(CarAnalysis).filter(CarAnalysis.success == True).count(),
                'total_market_data': session.query(MarketData).count(),
                'total_knowledge': session.query(KnowledgeBase).count(),
                'total_sessions': session.query(AnalysisSession).count()
            }

            # 计算成功率
            if stats['total_analyses'] > 0:
                stats['success_rate'] = stats['successful_analyses'] / stats['total_analyses'] * 100
            else:
                stats['success_rate'] = 0

            # 最近的分析
            recent_analysis = session.query(CarAnalysis).order_by(
                desc(CarAnalysis.created_at)
            ).first()

            if recent_analysis:
                stats['last_analysis'] = recent_analysis.created_at.isoformat()

            return stats

    def get_popular_makes(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取热门品牌统计"""
        with self.get_session() as session:
            results = session.query(
                Car.make,
                func.count(Car.id).label('count'),
                func.avg(Car.price_paid).label('avg_price')
            ).group_by(Car.make).order_by(desc('count')).limit(limit).all()

            return [{
                'make': result.make,
                'count': result.count,
                'avg_price': float(result.avg_price) if result.avg_price else 0
            } for result in results]