import os
import logging
import cv2
import torch
import numpy as np
from ultralytics import YOLO

logging.getLogger("ultralytics").setLevel(logging.ERROR)

class AI_Analyzer:
    def __init__(self, model_path="rsc/best.pt"):
        self.model_path = model_path
        self.model = None
        self.device = 'cpu'
        self.check_device()
        self.load_model()

    def check_device(self):
        """ 사용 가능한 장치 확인 (MPS > CUDA > CPU) """
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("🚀 [AI] Apple Silicon GPU (MPS) 가속을 사용합니다!")
        elif torch.cuda.is_available():
            self.device = 'cuda'
            print("🚀 [AI] NVIDIA GPU (CUDA) 가속을 사용합니다!")
        else:
            self.device = 'cpu'
            print("🐌 [AI] CPU를 사용합니다.")

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                print(f"🔄 [YOLO] 모델 로딩 중... ({self.model_path})")
                self.model = YOLO(self.model_path)
                # 모델을 해당 장치로 이동
                self.model.to(self.device) 
                print("✅ [YOLO] 모델 로드 완료")
            except Exception as e:
                print(f"❌ [YOLO] 로드 실패: {e}")
                self.model = None
        else:
            print(f"⚠️ [YOLO] 파일 없음: {self.model_path}")
            self.model = None

    def predict(self, image):
        """
        Return:
          1. tag: 결과 태그 (모델 없으면 None)
          2. image: 박스나 글씨가 그려진 이미지
        """
        # [핵심 로직] 모델이 없을 때
        if self.model is None:
            # 1. 원본 복사
            error_img = image.copy()
            # 2. 이미지 중앙에 빨간색으로 에러 메시지 쓰기
            h, w = error_img.shape[:2]
            cv2.putText(error_img, "MODEL NOT FOUND", (int(w/4), int(h/2)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            print("⚠️ [AI] 모델이 없어 분석을 중단합니다. (Tag: None)")
            
            # 3. 태그는 None(null), 이미지는 에러이미지 반환
            return None, error_img

        try:
            # 정상 추론 로직
            results = self.model(image, verbose=False, conf=0.8)
            result = results[0]
            result_img = result.plot() # 박스 그려진 이미지

            tag = "OK"

            return tag, result_img

        except Exception as e:
            print(f"❌ [YOLO] 예측 에러: {e}")
            return None, image # 에러 시에도 None 리턴