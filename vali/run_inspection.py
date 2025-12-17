import sys
import os
import cv2
import numpy as np
import datetime
from vali import config as cfg
from .algo_core import NutInspector
from .ai_inspector import AIInspector
from .db_manager import DataManager

# 결과 이미지 저장용 폴더 생성
if not os.path.exists(cfg.PROCESSED_DIR): os.makedirs(cfg.PROCESSED_DIR)

def draw_and_save(img, filename, save_folder, cv_data, ai_data, text, timestamp_str):
    """
    [수정됨] 
    1. 인자 개수 맞춤 (timestamp_str 추가)
    2. 화면에 NG/OK 텍스트 안 그림
    3. 파일명에 시간 추가해서 저장
    """
    if img is None: return ""
    res_img = img.copy()
    
    # 1. CV 결과 그리기 (삭제됨 - 빈칸)

    # 2. AI 결과 그리기 (박스 + 점수)
    if ai_data and ai_data['found']:
        for box in ai_data['boxes']:
            x1, y1, x2, y2, conf = box
            # 빨간 박스
            cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            # 점수 표시
            label_text = f"Rust {conf:.2f}"
            cv2.putText(res_img, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # 3. 텍스트 표시 (삭제됨)
    # cv2.putText(res_img, text, ...) <--- 이 줄을 지웠습니다.

    # 4. 파일 저장 (시간 포함된 이름 생성)
    name_only = os.path.splitext(filename)[0]
    ext = os.path.splitext(filename)[1]
    
    # 예: result_top_20251102123000.jpg
    save_name = f"result_{name_only}_{timestamp_str}{ext}"
    save_path = os.path.join(save_folder, save_name)
    
    try:
        cv2.imwrite(save_path, res_img)
        print(f"   🖼️ [이미지 저장] {save_path}")
        return save_path
    except Exception as e:
        print(f"   ❌ 이미지 저장 실패: {e}")
        return ""

def run_algorithm(top_path, bot_path):
    """
    [핵심 함수] 사진 2장을 받아 검사 -> 저장
    Return: 1 (성공), 0 (실패)
    """
    print(f"\n>>> [System] 알고리즘 시작: Top={top_path}, Bot={bot_path}")
    now = datetime.datetime.now()
    timestamp_file = now.strftime("%Y%m%d%H%M%S")      # 파일명용 (예: 20251102120000)
    timestamp_db = now.strftime("%Y-%m-%d %H:%M:%S")    # DB용 (예: 2025-11-02 12:00:00
    # 0. 파일 존재 여부 확인
    if not os.path.exists(top_path):
        print(f"❌ 실패: Top 사진이 없습니다 -> {top_path}")
        return 0
    
    # 1. 초기화
    try:
        inspector = NutInspector()
        ai_inspector = AIInspector()
        db_mgr = DataManager()
    except Exception as e:
        print(f"❌ 초기화 오류: {e}")
        return 0
    now = datetime.datetime.now()
    timestamp_file = now.strftime("%Y%m%d%H%M%S")    # 파일명용 (20251102...)
    timestamp_db = now.strftime("%Y-%m-%d %H:%M:%S")  # DB용 (2025-11-02...)

    # ==========================================
    # [Step 1] Top 이미지 처리
    # ==========================================
    img_top_raw = cv2.imread(top_path)
    if img_top_raw is None: return 0
    
    # A. AI 검사
    res_ai_top = ai_inspector.inspect(img_top_raw, "Top")

    # B. CV 검사
    img_top_calib = inspector.load_and_calibrate(top_path)
    data_cv = inspector.analyze(img_top_calib)
    
    res_cv = None
    if data_cv:
        angle = inspector.find_best_angle(data_cv)
        res_cv = inspector.inspect(data_cv, angle)
        
        # (!!!) [중요 수정] 데이터 상호 교환 (KeyError 방지)
        # 1. 그림 그릴 때 필요함: 1차 데이터(data_cv)에 구멍 정보(hole) 추가
        data_cv['hole'] = res_cv['hole']
        
        # 2. DB 저장할 때 필요함: 2차 데이터(res_cv)에 중심점 정보(center) 추가
        res_cv['center'] = data_cv['center']
        
    else:
        print("   ❌ CV 분석 실패 (너트 미검출)")
        return 0 

    # ==========================================
    # [Step 2] Bottom 이미지 처리
    # ==========================================
    img_bot_raw = None
    
    # (!!!) [수정 전] 이렇게 되어 있어서 에러가 났습니다.
    # res_ai_bot = {"found": False, "boxes": [], "score": 0.0, "res": "No Image"}
    
    # (!!!) [수정 후] 'score'를 'conf'로 바꿔주세요! (AI 모듈과 이름 통일)
    res_ai_bot = {"found": False, "boxes": [], "conf": 0.0, "res": "No Image"}
    
    if os.path.exists(bot_path):
        img_bot_raw = cv2.imread(bot_path)
        if img_bot_raw is not None:
            res_ai_bot = ai_inspector.inspect(img_bot_raw, "Bottom")
    
    # ==========================================
    # [Step 3] 결과 이미지 생성 및 저장
    # ==========================================
    temp_text = "NG" if (res_ai_top['found'] or res_ai_bot['found']) else "OK"
    if res_cv and (res_cv['shape']['res'] == "FAIL" or res_cv['hole']['res'] == "FAIL"): temp_text = "NG"

    # Top 저장 (data_cv에는 이제 hole 정보가 들어있으므로 에러 안 남)
    top_proc_path = draw_and_save(
        img_top_calib if img_top_calib is not None else img_top_raw, 
        os.path.basename(top_path), 
        cfg.RESULT_DIR_TOP, # results_top 폴더
        data_cv, 
        res_ai_top, 
        temp_text,
        timestamp_file # (!!!) 시간 전달
    )
    
    # Bottom 저장
    bot_proc_path = ""
    if img_bot_raw is not None:
        bot_proc_path = draw_and_save(
            img_bot_raw, 
            os.path.basename(bot_path), 
            cfg.RESULT_DIR_BOTTOM, # results_bottom 폴더
            None, 
            res_ai_bot, 
            "",
            timestamp_file # (!!!) 시간 전달
        )

    if not top_proc_path:
        print("❌ 결과 이미지 저장 실패")
        return 0

    # ==========================================
    # [Step 4] DB 저장
    # ==========================================
    area = data_cv['area'] if data_cv else 0
    
    try:
        # res_cv에는 이제 center 정보가 들어있으므로 에러 안 남
        sid, txt = db_mgr.save_result(res_cv, res_ai_top, res_ai_bot, area, top_proc_path, bot_proc_path,timestamp_db)
        print(f"✅ DB 저장 완료! (ID: {sid}) | 결과: {txt}")
        return 1  # 성공!
    except Exception as e:
        print(f"❌ DB 저장 실패: {e}")
        return 0  # 실패!


# =========================================================
# [실행부] 입력값 강제 확인 (2장 필수)
# =========================================================
if __name__ == "__main__":
    # (!!!) [수정] 인자가 3개(파일명 + top + bot)가 아니면 무조건 에러
    # sys.argv[0]은 실행 파일명(run_inspection.py)이므로, 총 길이가 3이어야 함
    if len(sys.argv) != 3:
        print("\n❌ 오류: 사진 파일 2개가 반드시 필요합니다!")
        print("👉 사용법: python run_inspection.py [Top사진] [Bottom사진]")
        print("   예시: python run_inspection.py top.jpg bottom.jpg")
        sys.exit(1) # 에러 코드 1 반환하며 종료

    t_path = sys.argv[1]
    b_path = sys.argv[2]

    # 파일 존재 여부 한 번 더 체크 (친절한 에러 메시지용)
    if not os.path.exists(t_path):
        print(f"❌ 오류: 첫 번째 파일(Top)이 없습니다 -> {t_path}")
        sys.exit(1)
        
    if not os.path.exists(b_path):
        print(f"❌ 오류: 두 번째 파일(Bottom)이 없습니다 -> {b_path}")
        sys.exit(1)

    # 실행
    result_code = run_algorithm(t_path, b_path)
    sys.exit(0 if result_code == 1 else 1)