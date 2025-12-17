import sqlite3
import json
import datetime
import numpy as np
from vali import config as cfg

class DataManager:
    def __init__(self):
        """
        [초기화] DB 매니저가 시작될 때 실행됩니다.
        config.py에 설정된 DB 파일 경로를 가져옵니다.
        """
        self.db_path = cfg.DB_FILE
        print(f"📂 DB 연결 주소: {self.db_path}")

    def save_result(self, cv_data, ai_top, ai_bot, area, cam1_path, cam2_path, timestamp):
        """
        [핵심 기능] 검사 결과를 DB에 저장합니다.
        
        1. 불량 사유를 '101' 같은 3자리 코드로 변환합니다.
        2. 픽셀(px) 단위가 아닌 밀리미터(mm) 단위 값을 저장합니다.
        """
        
        # (1) 제품 기준값(공차 등)을 DB 'Product' 테이블에 업데이트 (필수 절차)
        self.register_product()

        # DB 연결 시작
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # (2) Measurements 테이블이 없으면 새로 만듭니다. (안전 장치)
        # 컬럼 설명:
        # measured_center: 중심점 좌표, measured_contour: 외곽선 점들 (JSON 문자열)
        # area_size: 면적(mm2), hole_offset: 편심량(mm)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Measurements (
                measure_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                measured_at TEXT DEFAULT CURRENT_TIMESTAMP, 
                inspection_result TEXT, 
                cam1_path TEXT, 
                cam2_path TEXT, 
                measured_center TEXT, 
                product_id INTEGER, 
                measured_contour TEXT, 
                model_score REAL, 
                hole_offset REAL, 
                area_size REAL, 
                fail_reason TEXT, 
                FOREIGN KEY (product_id) REFERENCES Product (product_id)
            )
        ''')
        
        # --- [Logic 1] 3자리 불량 코드 생성 ("000" ~ "111") ---
        # 기본값은 "0" (정상)으로 둡니다.
        code_shape = "0"  # 첫째 자리: 외곽선 형상
        code_hole = "0"   # 둘째 자리: 구멍 편심
        code_rust = "0"   # 셋째 자리: 녹(Rust)

        # 1. CV 분석 결과가 있을 때만 형상/구멍 불량을 판단합니다.
        if cv_data:
            if cv_data['shape']['res'] == "FAIL": 
                code_shape = "1"
            if cv_data['hole']['res'] == "FAIL": 
                code_hole = "1"
            
        # 2. AI 분석 결과 (녹) 판단
        # 상부(Top)나 하부(Bot) 중 하나라도 NG가 나오면 불량 처리
        if ai_top['res'] == "NG" or ai_bot['res'] == "NG":
            code_rust = "1"
            
        # 3. 코드 조합 (예: 외곽선불량+녹불량 = "101")
        fail_code = f"{code_shape}{code_hole}{code_rust}"
        
        # 4. 최종 판정 (코드에 '1'이 하나라도 있으면 NG)
        final_res = "NG" if "1" in fail_code else "OK"
        reason = fail_code  # DB에 저장될 사유는 이제 "101" 같은 코드입니다.

        # --- [Logic 2] mm 단위 데이터 추출 및 JSON 변환 ---
        if cv_data:
            # algo_core.py에서 계산된 mm 단위 값을 가져옵니다.
            # .get()을 쓰는 이유: 혹시라도 키가 없을 때 에러가 나지 않고 0.0을 넣기 위함입니다.
            real_area = cv_data.get('area_mm2', 0.0)      # 면적 (mm^2)
            real_offset = cv_data['hole'].get('offset_mm', 0.0) # 편심량 (mm)
            
            # 외곽선 좌표 (그래프 그리기용이므로 픽셀 단위 유지) -> JSON 문자열로 변환
            # tolist(): numpy 배열은 DB에 바로 저장이 안 되어서 리스트로 바꿉니다.
            contour_json = json.dumps({
                "x": cv_data['shape']['x'].tolist(), 
                "y": cv_data['shape']['y'].tolist()
            })
            
            # 중심점 및 기타 상세 정보 정리
            hole = cv_data['hole']
            idx = np.argmax(cv_data['shape']['y']) # 가장 먼 점 찾기용 인덱스
            
            center_data = {
                "hole_found": hole['found'],
                "hole_cx": hole['rot_center'][0] if hole['found'] else 0,
                "hole_cy": hole['rot_center'][1] if hole['found'] else 0,
                # farthest: 중심에서 가장 먼 점 (형상 오차 확인용)
                "farthest_x": cv_data['shape']['x'][idx] if len(cv_data['shape']['x']) > 0 else 0,
                "farthest_y": cv_data['shape']['y'][idx] if len(cv_data['shape']['y']) > 0 else 0,
                # body: 너트 몸통의 중심점 (run_inspection에서 넘겨받음)
                "body_cx": cv_data['center'][0] if cv_data.get('center') else 0,
                "body_cy": cv_data['center'][1] if cv_data.get('center') else 0
            }
            center_json = json.dumps(center_data)
            
        else:
            # 분석 실패 시 기본값(0)으로 채웁니다.
            real_area = 0.0
            real_offset = 0.0
            contour_json = "{}"
            center_json = "{}"

        # AI 확신도 점수 (상/하부 중 더 높은 점수를 저장)
        final_ai_score = max(ai_top['conf'], ai_bot['conf'])

        # --- [Logic 3] DB에 최종 저장 (INSERT) ---
        # 여기서 float(real_area)가 들어가면서 mm 단위 값이 저장됩니다.
        cursor.execute('''
            INSERT INTO Measurements 
            (product_id, measured_at, inspection_result, fail_reason, cam1_path, cam2_path,
             measured_center, measured_contour, area_size, hole_offset, model_score)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, final_res, reason, cam1_path, cam2_path, 
              center_json, contour_json, 
              float(real_area),    # [저장] 면적 (mm^2)
              float(real_offset),  # [저장] 편심량 (mm)
              float(final_ai_score)))
        
        # 저장된 행의 ID(번호)를 가져옵니다. (로그 출력용)
        lid = cursor.lastrowid
        conn.commit() # 저장 확정
        conn.close()  # 연결 종료
        
        return lid, reason

    def register_product(self, product_id=1, name="Hex Nut M6"):
        """
        [설정 동기화] config.py의 기준값들을 DB 'Product' 테이블에 저장합니다.
        (나중에 뷰어에서 기준선을 그릴 때 사용됩니다)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS Product (
                product_id INTEGER PRIMARY KEY, product_name TEXT, template_data TEXT, 
                tol_shape REAL, tol_hole REAL, limit_warn REAL, limit_fail REAL
            )
        ''')
        # 정답 템플릿 모양도 JSON으로 저장
        template_json = json.dumps({"x": cfg.TEMPLATE_X.tolist(), "y": cfg.TEMPLATE_Y.tolist()})
        
        cursor.execute('''
            INSERT OR REPLACE INTO Product 
            (product_id, product_name, template_data, tol_shape, tol_hole, limit_warn, limit_fail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (product_id, name, template_json, cfg.TOL_SHAPE, cfg.TOL_HOLE, cfg.LIMIT_WARNING, cfg.LIMIT_FAIL))
        conn.commit(); conn.close()

    def load_result(self, db_id):
        """
        [데이터 로드] 뷰어(Visualizer)가 요청한 ID의 검사 결과를 DB에서 꺼내줍니다.
        (저장 로직이 바뀌어도, 읽는 로직은 DB 값을 그대로 가져오므로 수정할 필요가 없습니다)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row # 데이터를 딕셔너리처럼 이름으로 꺼내기 위해 설정
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM Measurements WHERE measure_id=?", (db_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row: return None
        
        # JSON 문자열로 저장된 좌표들을 다시 파이썬 리스트/딕셔너리로 복구
        try:
            contour = json.loads(row['measured_contour']) if row['measured_contour'] else {}
            center = json.loads(row['measured_center']) if row['measured_center'] else {}
        except:
            contour, center = {}, {}

        # 뷰어가 쓰기 좋은 형태로 포장해서 반환
        return {
            "result_text": row['fail_reason'], # 예: "101"
            "area": row['area_size'],          # 예: 315.5 (mm2)
            "hole_offset": row['hole_offset'], # 예: 0.5 (mm)
            "measured_x": contour.get('x', []), "measured_y": contour.get('y', []),
            "hole_found": center.get('hole_found', False),
            "hole_cx": center.get('hole_cx', 0), "hole_cy": center.get('hole_cy', 0),
            "farthest_x": center.get('farthest_x', 0), "farthest_y": center.get('farthest_y', 0),
            "hole_x": [], "hole_y": [], 
            "rust_top": {}, "rust_bot": {},
            "cam1_path": row['cam1_path'],
            "cam2_path": row['cam2_path'],
            "template_x": cfg.TEMPLATE_X, "template_y": cfg.TEMPLATE_Y,
            "tol_shape": cfg.LIMIT_FAIL, "tol_hole": cfg.TOL_HOLE, "limit_warn": cfg.LIMIT_WARNING
        }