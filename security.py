import hashlib
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import jwt,JWTError

# 설정
SECRET_KEY = "super-secret-key" # 실제로는 복잡하게 설정하세요
ALGORITHM = "HS256"

# bcrypt 설정
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Hash:
    @staticmethod
    def _pre_hash(password: str) -> str:
        """
        [해결책] 72바이트 제한을 피하기 위해 먼저 SHA256으로 해싱합니다.
        이렇게 하면 길이에 상관없이 항상 64글자(Hex)가 나옵니다.
        """
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    @staticmethod
    def bcrypt(password: str):
        """ 비밀번호 암호화 (SHA256 -> bcrypt) """

        pre_hashed_pw = Hash._pre_hash(password)

        return pwd_context.hash(pre_hashed_pw)

    @staticmethod
    def verify(plain_password, hashed_password):
        """ 비밀번호 검증 """
        pre_hashed_pw = Hash._pre_hash(plain_password)
        return pwd_context.verify(pre_hashed_pw, hashed_password)
    
    # [추가됨] 토큰 생성 함수
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # 유효기간을 안 넣으면 기본적으로 100년(36500일)으로 설정 -> 사실상 무제한
        expire = datetime.utcnow() + timedelta(days=36500)
    
    # exp 클레임 추가
    to_encode.update({"exp": expire})
    
    # 암호화해서 문자열(토큰) 리턴
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload # 성공하면 데이터(user_id 등) 리턴
    except JWTError:
        return None 