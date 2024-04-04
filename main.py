from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import sessionmaker
from gensim.models import FastText
from dotenv import load_dotenv
import os
from typing import List
from sqlalchemy import ForeignKey
from sqlalchemy import or_
from sqlalchemy import Column, Integer, String, DateTime, Text, Float 
from sqlalchemy.orm import declarative_base
from datetime import datetime

# FastText 모델 로드
# 이 경로는 사전 훈련된 FastText 모델 파일의 실제 경로로 대체해야 합니다.
print('fasttext_load_start')
load_dotenv()
DB_PW = os.environ.get('DB_PW')

fasttext_model = FastText.load("data/fasttext_vector2000_neg100.bin")
print('fasttext_loaded')
# 데이터베이스 설정df 
DATABASE_URL = f"mysql+aiomysql://wooksbaby:{DB_PW}@localhost/ctee"
engine = create_async_engine(DATABASE_URL)
SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()

# 모델 정의
Base = declarative_base()

class Item(Base):
    __tablename__ = 'items'
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey('stores.store_id'))
    content = Column(Text)
    simple_contents = Column(Text)
    content = Column(Text)
    story = Column(Text)
    tags = Column(String(255))
    view_count = Column(Integer, default=0)

class Store(Base):
    __tablename__ = 'stores'
    store_id = Column(Integer, primary_key=True)
    title = Column(String(255))
    content = Column(Text)
    alias = Column(String(255))
    view_count = Column(Integer, default=0)
    sales_volume = Column(Integer)
    donation_count = Column(Integer)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    tags = Column(String(255))

class Event(Base):
    __tablename__ = 'event_projects'
    id = Column(Integer, primary_key=True)
    store_id = Column(Integer, ForeignKey('stores.store_id'))
    title = Column(String(255))
    simple_description = Column(Text)
    story = Column(Text)
    categories_event_id = Column(Integer)
    currency_code = Column(String(3))
    price = Column(Float)
    discount_price = Column(Float, nullable=True)
    discount_started_at = Column(DateTime, nullable=True)
    discount_ended_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)
    
# 비동기 데이터베이스 세션 의존성
async def get_db():
    async with SessionLocal() as session:
        yield session

# FastText를 이용하여 검색어와 유사한 단어를 찾는 함수
def get_similar_words(query: str, topn: int = 5) -> List[str]:
    similar_words = fasttext_model.wv.most_similar(query, topn=topn)
    # get_nearest_neighbors 함수는 (유사도, 단어) 튜플을 반환하므로 단어만 추출
    
    print(similar_words)
    return [word for word, _ in similar_words]

# 검색어와 관련된 항목을 데이터베이스에서 검색하는 엔드포인트
@app.get("/")
async def root():
    return {"message": "Welcome to my FastAPI application!"}

@app.get("/search/{query}")
async def search(query: str, db: AsyncSession = Depends(get_db)):
    # 검색어 확장
    expanded_queries = get_similar_words(query)
    
    # Items 검색
    items_result = await search_items(query, expanded_queries, db)
    
    # Stores 검색
    stores_result = await search_stores(query, expanded_queries, db)
    
    # Events 검색
    events_result = await search_events(query, expanded_queries, db)

    return {
        "items": items_result,
        "stores": stores_result,
        "events": events_result,
        "expend": expanded_queries
    }
async def search_items(original_query, expanded_queries, db):
    # 원래 검색어로 'simple_contents', 'content', 'story', 'tags'에서 검색
    original_query_result = await db.execute(select(Item).where(
        or_(
            Item.simple_contents.contains(original_query),
            Item.content.contains(original_query),
            Item.story.contains(original_query),
            Item.tags.contains(original_query)
        )
    ).order_by(Item.view_count.desc()))
    
    # 확장된 검색어로 'tags'에서만 검색
    expanded_query_result = []
    for query in expanded_queries:
        result = await db.execute(select(Item).where(Item.tags.contains(query)).order_by(Item.view_count.desc()))
        expanded_query_result.extend(result.scalars().all())
    
    # 결과 합치기 및 중복 제거
    items_result = list(set(original_query_result.scalars().all() + expanded_query_result))
    # 조회수 순으로 정렬
    items_result.sort(key=lambda x: x.view_count, reverse=True)
    return items_result[:10]

async def search_stores(original_query, expanded_queries, db):
    # 원본 검색어로 'title', 'tags'에서 검색
    original_query_result = await db.execute(select(Store).where(
        or_(
            Store.title.contains(original_query),
            Store.tags.contains(original_query)
        )
    ).order_by(Store.view_count.desc()))
    
    # 확장된 검색어로 'tags'에서만 검색
    expanded_query_result = []
    for query in expanded_queries:
        result = await db.execute(select(Store).where(Store.tags.contains(query)).order_by(Store.view_count.desc()))
        expanded_query_result.extend(result.scalars().all())
    
    # 결과 합치기 및 중복 제거
    stores_result = list(set(original_query_result.scalars().all() + expanded_query_result))
    # 조회수 순으로 정렬
    stores_result.sort(key=lambda x: x.view_count, reverse=True)
    return stores_result[:10]

async def search_events(original_query, expanded_queries, db):
    # 원본 및 확장된 검색어로 'title'에서 검색
    query_set = [original_query] + expanded_queries
    events_result = []
    for query in query_set:
        result = await db.execute(select(Event).where(Event.title.contains(query)).order_by(Event.updated_at.desc()))
        events_result.extend(result.scalars().all())
    
    # 중복 제거
    unique_events = list({event.id: event for event in events_result}.values())
    return unique_events[:10]
print('done!')