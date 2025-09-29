"""Database models for car analysis system with RAG support"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional, Dict, Any

Base = declarative_base()


class Car(Base):
    """汽车基本信息表"""
    __tablename__ = 'cars'

    id = Column(Integer, primary_key=True, autoincrement=True)
    make = Column(String(50), nullable=False, index=True)  # 品牌
    model = Column(String(100), nullable=False, index=True)  # 型号
    year = Column(Integer, nullable=False, index=True)  # 年份
    mileage = Column(Integer, nullable=False)  # 里程
    price_paid = Column(Float, nullable=False)  # 成交价格
    trim = Column(String(100))  # 配置版本
    color = Column(String(50))  # 颜色
    transmission = Column(String(50))  # 变速箱
    engine = Column(String(100))  # 发动机
    fuel_type = Column(String(50))  # 燃料类型
    condition = Column(String(50))  # 车况
    location = Column(String(100))  # 地区

    # PDF来源信息
    pdf_source = Column(String(500))  # PDF文件路径
    pdf_page = Column(Integer)  # PDF页码
    raw_text = Column(Text)  # 原始文本

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关联关系
    analyses = relationship("CarAnalysis", back_populates="car")
    market_data = relationship("MarketData", back_populates="car")


class CarAnalysis(Base):
    """汽车分析结果表"""
    __tablename__ = 'car_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)

    # 分析结果
    rule_based_score = Column(Integer)  # 规则评分 0-100
    rule_based_verdict = Column(String(50))  # 规则评价
    llm_score = Column(Integer)  # LLM评分 0-100
    llm_verdict = Column(String(50))  # LLM评价
    llm_reasoning = Column(Text)  # LLM推理过程

    # 价格分析
    market_median_price = Column(Float)  # 市场中位价
    price_delta = Column(Float)  # 价格差异
    price_delta_percent = Column(Float)  # 价格差异百分比
    deal_category = Column(String(50))  # 交易类型

    # 数据来源
    data_source = Column(String(100))  # 数据来源
    comparable_count = Column(Integer)  # 可比车辆数量
    research_quality = Column(String(50))  # 研究质量

    # 分析状态
    success = Column(Boolean, default=False)
    error_message = Column(Text)
    analysis_version = Column(String(20))  # 分析版本

    # 完整分析数据（JSON格式）
    full_analysis_data = Column(JSON)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    car = relationship("Car", back_populates="analyses")


class MarketData(Base):
    """市场数据表 - 存储搜索到的可比车辆信息"""
    __tablename__ = 'market_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    car_id = Column(Integer, ForeignKey('cars.id'), nullable=False)

    # 搜索信息
    search_query = Column(String(500))  # 搜索查询
    search_engine = Column(String(50), default='tavily')  # 搜索引擎

    # 找到的可比车辆
    comparable_make = Column(String(50))
    comparable_model = Column(String(100))
    comparable_year = Column(Integer)
    comparable_mileage = Column(Integer)
    comparable_price = Column(Float)
    comparable_url = Column(String(1000))  # 原始链接
    comparable_source = Column(String(100))  # 数据来源网站

    # 相似度评分
    similarity_score = Column(Float)  # 与目标车辆的相似度

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关联关系
    car = relationship("Car", back_populates="market_data")


class AnalysisSession(Base):
    """分析会话表 - 记录每次PDF分析会话"""
    __tablename__ = 'analysis_sessions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), unique=True, nullable=False)  # 会话ID

    # 会话信息
    pdf_path = Column(String(500))
    pdf_content = Column(Text)  # PDF完整内容
    cars_extracted = Column(Integer, default=0)  # 提取的车辆数
    cars_analyzed = Column(Integer, default=0)  # 分析成功的车辆数

    # 会话结果
    success_rate = Column(Float)  # 成功率
    average_score = Column(Float)  # 平均评分
    total_errors = Column(Integer, default=0)

    # 会话配置
    analysis_config = Column(JSON)  # 分析配置

    # 时间戳
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    # 报告文件路径
    report_path = Column(String(500))


class KnowledgeBase(Base):
    """知识库表 - 存储汽车相关知识用于RAG"""
    __tablename__ = 'knowledge_base'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 知识内容
    title = Column(String(200), nullable=False)
    content = Column(Text, nullable=False)
    content_type = Column(String(50))  # 内容类型：car_info, market_analysis, pricing_guide等

    # 分类标签
    category = Column(String(100))  # 大类：brand, model, market_trends等
    subcategory = Column(String(100))  # 小类
    tags = Column(JSON)  # 标签数组

    # 相关车辆信息（如果适用）
    related_make = Column(String(50))
    related_model = Column(String(100))
    related_year_start = Column(Integer)
    related_year_end = Column(Integer)

    # 数据来源
    source = Column(String(200))  # 数据来源
    source_url = Column(String(1000))  # 原始链接
    reliability_score = Column(Float, default=1.0)  # 可靠性评分

    # 使用统计
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class UserQuery(Base):
    """用户查询表 - 记录用户的问题和回答，用于改进RAG"""
    __tablename__ = 'user_queries'

    id = Column(Integer, primary_key=True, autoincrement=True)

    # 查询信息
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50))  # 查询类型
    user_id = Column(String(100))  # 用户标识

    # RAG检索结果
    retrieved_docs = Column(JSON)  # 检索到的文档
    retrieval_score = Column(Float)  # 检索质量评分

    # 回答信息
    response_text = Column(Text)
    response_quality = Column(String(50))  # 回答质量
    user_feedback = Column(String(50))  # 用户反馈：helpful, not_helpful

    # 关联分析
    related_car_id = Column(Integer, ForeignKey('cars.id'))
    related_analysis_id = Column(Integer, ForeignKey('car_analyses.id'))

    # 时间戳
    created_at = Column(DateTime, default=datetime.utcnow)


# 数据库操作助手类
class DatabaseHelper:
    """数据库操作助手"""

    @staticmethod
    def create_tables(engine):
        """创建所有表"""
        Base.metadata.create_all(engine)

    @staticmethod
    def car_to_dict(car: Car) -> Dict[str, Any]:
        """将Car对象转换为字典"""
        return {
            'id': car.id,
            'make': car.make,
            'model': car.model,
            'year': car.year,
            'mileage': car.mileage,
            'price_paid': car.price_paid,
            'trim': car.trim,
            'color': car.color,
            'transmission': car.transmission,
            'engine': car.engine,
            'fuel_type': car.fuel_type,
            'condition': car.condition,
            'location': car.location,
            'created_at': car.created_at.isoformat() if car.created_at else None
        }

    @staticmethod
    def analysis_to_dict(analysis: CarAnalysis) -> Dict[str, Any]:
        """将CarAnalysis对象转换为字典"""
        return {
            'id': analysis.id,
            'car_id': analysis.car_id,
            'rule_based_score': analysis.rule_based_score,
            'rule_based_verdict': analysis.rule_based_verdict,
            'llm_score': analysis.llm_score,
            'llm_verdict': analysis.llm_verdict,
            'llm_reasoning': analysis.llm_reasoning,
            'market_median_price': analysis.market_median_price,
            'price_delta': analysis.price_delta,
            'price_delta_percent': analysis.price_delta_percent,
            'deal_category': analysis.deal_category,
            'data_source': analysis.data_source,
            'comparable_count': analysis.comparable_count,
            'success': analysis.success,
            'created_at': analysis.created_at.isoformat() if analysis.created_at else None
        }