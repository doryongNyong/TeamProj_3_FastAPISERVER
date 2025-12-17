from fastapi import APIRouter, Depends, HTTPException, Request, status, Response
from sqlalchemy.orm import Session
from pydantic import BaseModel
from models import User, get_db
from security import Hash, create_access_token, verify_token

# API 주소 프리픽스 (/api/login이 됨)
router = APIRouter(prefix="/api", tags=["Auth"])

# --- 데이터 규격 (Schemas) ---
class UserSignup(BaseModel):
    id: str
    pw: str
    name: str
    role: int

class UserLogin(BaseModel):
    id: str
    pw: str

# --- [1] 회원가입 ---
@router.post("/signup")
def signup(request: UserSignup, db: Session = Depends(get_db)):
    if db.query(User).filter(User.login_id == request.id).first():
        raise HTTPException(status_code=401, detail="이미 존재하는 ID입니다.")

    new_user = User(
        login_id=request.id,
        password_hash=Hash.bcrypt(request.pw),
        user_name=request.name,
        role=request.role # User 권한
    )
    db.add(new_user)
    db.commit()
    
    return {
        "status" : "success",
        "code" : 200,
        "message": "회원가입 완료", 
        "user_id": request.id}

# --- [2] 로그인 (수정됨) ---
@router.post("/login")
def login(login_data: UserLogin,response: Response ,db: Session = Depends(get_db)):
    # 1. 유저 확인
    user = db.query(User).filter(User.login_id == login_data.id).first()
    
    if not user or not Hash.verify(login_data.pw, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_UNAUTHORIZED,
            detail="아이디 또는 비밀번호 오류"
        )

    # 2. [핵심] 무제한 토큰 생성
    access_token = create_access_token(
        data={"sub": user.login_id,"name":user.user_name, "role": user.role}
    )

    #3.
    response.set_cookie(
        key="access_token",      # 쿠키 이름
        value=access_token,      # 쿠키 값 (JWT)
        secure=False,            
        samesite="Lax"           
    )

    # 4. 
    return {
        "status": "success",
        "code" : 200,
        "token_type": "bearer",
        "name": user.user_name,
        "id": user.login_id,
        "role": user.role
    }

# --- [3] 로그아웃 ---
@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(key="access_token")
    
    return {"message": "로그아웃 성공", "status": "success", "code": 200}

# --- [4] 상태 확인 ---
@router.get("/me")
def me(request: Request):
    token = request.cookies.get("access_token")
    

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="로그인이 필요합니다 (토큰 없음)."
        )
    
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않거나 만료된 토큰입니다."
        )
        
    return {
        "status": "LoggedIn",
        "user_id": payload.get("uid"),   # 유저 PK
        "login_id": payload.get("sub"),  # 로그인 ID
        "name": payload.get("name"),     # 이름
        "role": payload.get("role")      # 권한 (1 or 2)
    }