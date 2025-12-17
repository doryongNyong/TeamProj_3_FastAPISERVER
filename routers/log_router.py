import base64
import os

from fastapi import APIRouter, Depends, HTTPException, Request, status, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, case, and_
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from models import Product, Measurement, get_db

# API 주소 프리픽스 (/api/login이 됨)
router = APIRouter(prefix="/api", tags=["LOG"])

class LogRequest(BaseModel):
    startDate: str

class LogResponse(BaseModel):
    mid: int
    timestamp: Optional[str] # datetime을 문자열로 변환해서 줄 수도 있음
    product_name: str
    result: str

class ImageResponse(BaseModel):
    img1_base64: Optional[str]
    img2_base64: Optional[str]

class LogDetailRequest(BaseModel):
    mid: int

class LogDetailResponse(BaseModel):
    measure_id: int
    measured_at: Optional[datetime]
    inspection_result: Optional[str]
    measured_center: Optional[str]
    product_id: Optional[int]
    measured_contour: Optional[str]
    model_score: Optional[float]
    hole_offset: Optional[float]
    area_size: Optional[float]
    fail_reason: Optional[str]
    
    # ORM 객체(SQLAlchemy model)를 Pydantic 모델로 변환 허용
    class Config:
        from_attributes = True

# [요청] 기간 조회 데이터
class StatisticsRequest(BaseModel):
    startDate: str  # "yyyy-MM-dd"
    endDate: str    # "yyyy-MM-dd"

# [응답] 일별 통계 아이템
class DailyStatItem(BaseModel):
    date: str
    total: int
    defect: int

# [응답] 불량 유형별 카운트
class DefectCountItem(BaseModel):
    shape: int
    center: int
    rust: int
    total_ng: int

# [응답] 전체 통계 응답 (C# ServerStats와 매핑)
class StatisticsResponse(BaseModel):
    daily_data: List[DailyStatItem]
    counts: DefectCountItem

@router.post("/logs", response_model=List[LogResponse])
def get_logs(req: LogRequest, db: Session = Depends(get_db)):
    logs = db.query(Measurement).join(Product)\
             .filter(Measurement.measured_at.like(f"{req.startDate}%"))\
             .order_by(Measurement.measure_id.desc())\
             .all()

    results = []
    for log in logs:
        p_name = log.product.product_name if log.product else "Unknown"
        measured_date_str = ""
        if log.measured_at:

            if hasattr(log.measured_at, 'strftime'):
                measured_date_str = log.measured_at.strftime("%Y-%m-%d")

            else:
                measured_date_str = str(log.measured_at)[:10]

        results.append(LogResponse(
            mid=log.measure_id,
            timestamp=measured_date_str,
            product_name=p_name,
            result=log.inspection_result if log.inspection_result else ""
        ))
    return results

# --- [6] 이미지 상세 조회 (WPF 연동용) ---
@router.get("/logs/{mid}/images", response_model=ImageResponse)
def get_log_images(mid: int, db: Session = Depends(get_db)):
    log = db.query(Measurement).filter(Measurement.measure_id == mid).first()
    
    if not log:
        raise HTTPException(status_code=400, detail="Log not found")

    cam1_clean = log.cam1_path.replace("\\", "/") if log.cam1_path else None
    cam2_clean = log.cam2_path.replace("\\", "/") if log.cam2_path else None

    def encode_img_from_path(path_str):
        """파일 경로에서 이미지를 읽어 Base64 문자열로 인코딩."""
        if not path_str or not os.path.exists(path_str):
            return None
        
        try:
            with open(path_str, "rb") as f:
                img_data = f.read()
            return base64.b64encode(img_data).decode('utf-8')
        except Exception:
            return None 

    return ImageResponse(
        img1_base64=encode_img_from_path(os.path.join("results",cam1_clean)),
        img2_base64=encode_img_from_path(os.path.join("results",cam2_clean))
    )

@router.get("/logsdetail", response_model=LogDetailResponse)
def get_log_detail(mid: int, db: Session = Depends(get_db)):
    """
    특정 로그(mid)의 상세 정보를 반환합니다. (이미지 경로는 제외)
    """
    log = db.query(Measurement).filter(Measurement.measure_id == mid).first()
    
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
        
    # Pydantic 모델(LogDetailResponse)이 알아서 cam1_path, cam2_path를 걸러내고
    # 정의된 필드만 JSON으로 변환해줍니다.
    return log
    
@router.post("/statistics", response_model=StatisticsResponse)
def get_statistics(req: StatisticsRequest, db: Session = Depends(get_db)):
    """
    선택한 기간(startDate ~ endDate) 동안의:
    1. 일별 검사 수량 및 불량 수량 (그래프용)
    2. 불량 유형별(외곽선, 중심, 녹) 합계 (카드용)
    """
    
    # 1. 날짜 범위 필터링 조건 생성
    # 문자열 비교를 위해 입력된 날짜 포맷 활용 (DB에 저장된 포맷이 "YYYY-MM-DD HH:MM:SS"라고 가정)
    start_dt = req.startDate + " 00:00:00"
    end_dt = req.endDate + " 23:59:59"
    
    date_col = func.strftime('%Y-%m-%d', Measurement.measured_at).label("date")
    
    daily_stats = db.query(
        date_col,
        func.count(Measurement.measure_id).label("total"),
        func.sum(case((Measurement.inspection_result == 'NG', 1), else_=0)).label("defect")
    ).filter(
        and_(Measurement.measured_at >= start_dt, Measurement.measured_at <= end_dt)
    ).group_by(
        date_col
    ).order_by(
        date_col.asc()
    ).all()

    daily_data_list = []
    for row in daily_stats:
        daily_data_list.append(DailyStatItem(
            date=row.date,
            total=row.total,
            defect=row.defect if row.defect else 0
        ))

    # =========================================================
    # [Query 2] 불량 유형별 집계 (Defect Counts) - 하단 카드용
    # =========================================================
    # NG 항목만 조회하여 fail_reason을 분석합니다.
    ng_logs = db.query(Measurement.fail_reason).filter(
        and_(
            Measurement.measured_at >= start_dt, 
            Measurement.measured_at <= end_dt,
            Measurement.inspection_result == 'NG'
        )
    ).all()

    count_shape = 0
    count_center = 0
    count_rust = 0
    total_ng = len(ng_logs)

    # fail_reason 문자열을 분석하여 카운팅 (C# 로직에 맞춰 키워드 분류)
    # 실제 DB에 저장되는 fail_reason 텍스트에 맞춰 키워드를 수정하세요.
    for log in ng_logs:
        reason = (log.fail_reason or "").upper()
        
        if "SHAPE" in reason or "CONTOUR" in reason or "SIZE" in reason or "WIDTH" in reason:
            count_shape += 1
        elif "CENTER" in reason or "HOLE" in reason or "OFFSET" in reason:
            count_center += 1
        elif "RUST" in reason or "STAIN" in reason or "COLOR" in reason:
            count_rust += 1
        else:
            # 기타 불량은 가장 가까운 유형이나 별도 처리가 필요하면 추가
            # 현재는 분류되지 않은 경우 shape로 퉁치거나 무시할 수 있음
            count_shape += 1 # 기본값 처리 예시

    counts_data = DefectCountItem(
        shape=count_shape,
        center=count_center,
        rust=count_rust,
        total_ng=total_ng
    )

    return StatisticsResponse(
        daily_data=daily_data_list,
        counts=counts_data
    )