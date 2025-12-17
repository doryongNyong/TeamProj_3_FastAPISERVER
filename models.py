from sqlalchemy import create_engine, Column, Integer, Float, String, Text , ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
import os

# --- [1] 데이터베이스 연결 설정 ---

# 현재 파일 위치 기준 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "factory.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# 엔진 생성
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

# 세션 및 Base 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 세션 의존성 함수 (FastAPI에서 Depends로 사용)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- [2] 테이블 정의 ---

class User(Base):
    # [중요] 스크린샷에 나온 테이블명 그대로 사용
    __tablename__ = "User"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    user_name = Column(String, nullable=False)
    login_id = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(Integer, default=1)
    


class Product(Base):
    __tablename__ = "Product"

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    product_name = Column(String, nullable=False)
    width = Column(Float)
    length = Column(Float)
    ref_center_point = Column(Float)
    ref_contour = Column(Float)
    template_data = Column(Text) # JSON 문자열 저장 가능
    limit_fail = Column(Float)
    limit_warn = Column(Float)
    tol_hole = Column(Float)
    tol_shape = Column(Float)

    # Measurements 테이블과의 관계 설정 (1:N)
    measurements = relationship("Measurement", back_populates="product")

# 4. 측정 결과 테이블 (추가됨)
class Measurement(Base):
    __tablename__ = "Measurements" # 테이블 이름 'Measurements'

    measure_id = Column(Integer, primary_key=True, autoincrement=True)
    # default=func.now()를 써서 생성 시점 자동 저장
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    
    inspection_result = Column(String) # OK/NG
    cam1_path = Column(String)
    cam2_path = Column(String)
    measured_center = Column(String) # "(x,y)" 문자열로 저장
    
    # FK: Product 테이블 참조
    product_id = Column(Integer, ForeignKey("Product.product_id"))
    
    measured_contour = Column(String)
    model_score = Column(Float)
    hole_offset = Column(Float)
    area_size = Column(Float)
    fail_reason = Column(String)

    # Product와의 관계 설정
    product = relationship("Product", back_populates="measurements")