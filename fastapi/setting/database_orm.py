"""동기 방식 DB ORM 연결 정의"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv # 추가

# .env 파일 내용을 환경 변수로 로드
load_dotenv()

#host=os.getenv("DB_HOST")
host = os.getenv("DB_HOST")
port=os.getenv("DB_PORT")
user=os.getenv("DB_USER")
password=os.getenv("DB_PASSWORD")
database=os.getenv("DB_NAME")
DATABASE_URL = f"postgresql://{user}:{password}@{host}:{port}/{database}"

engine = create_engine(DATABASE_URL, pool_size=10,max_overflow=20,pool_timeout=30,pool_recycle=1800)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def db_connection_pool():
    db = SessionLocal()
    try:
        print(">>>> DataBase ORM Setting Start  <<<<")
        yield db # db 연결에 성공한 경우 DB세션 시작
    finally:
        db.close()
        # db session 마무리 후 close

# def check_connection():
#     try:
#         # 엔진에서 직접 연결을 시도해 봅니다.
#         with engine.connect() as connection:
#             result = connection.execute(text("SELECT 1"))
#             print(">>>> ✅ DB 연결 성공! (테스트 쿼리 결과: 1) <<<<")
#     except SQLAlchemyError as e:
#         print(f">>>> ❌ DB 연결 실패: {e}")

# if __name__ == "__main__":
#     check_connection()