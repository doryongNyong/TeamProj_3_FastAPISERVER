import os
import numpy as np


# 파일 경로 기본값

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR,"data","factory.db")

# 1. 파일 경로 설정 (경로 결합)
# 이제 어디서 실행하든 무조건 /app/cvalgo/calibration_data.npz를 가리킵니다.
CALIB_FILE = os.path.join(BASE_DIR, "data", "calibration_data.npz")
AI_MODEL_PATH = os.path.join(BASE_DIR,"rsc","best.pt")

# 2. 자동화 폴더 설정
WATCH_DIR = "incoming"
PROCESSED_DIR = "processed"
RESULT_DIR_TOP = "/static/results_top"
RESULT_DIR_BOTTOM = "/static/results_bottom"
# 검사 기준
APPROX_EPSILON = 0.0001
CROP_MARGIN = 20
AI_CONF_THRES = 0.5
# [단위 변환] 1mm당 픽셀 수 (실측값)
PIXELS_PER_MM = 2.523

# 정답 템플릿 (12시 기준 시계방향)
TEMPLATE_X = np.array([  0,  55,  55,   0, -55, -55 ])
TEMPLATE_Y = np.array([ 63,  33, -33, -63, -33,  33 ])

# 공차 판정 기준
TOL_SHAPE = 5.0
TOL_HOLE = 5.0
LIMIT_WARNING = 4.5
LIMIT_FAIL = 6.0