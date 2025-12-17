import cv2
import numpy as np
from vali import config as cfg

class NutInspector:
    def __init__(self):
        """
        [초기화 단계]
        검사를 시작하기 전에 필요한 '기준'과 '도구'를 준비합니다.
        """
        # 1. 정답 템플릿 준비
        # config.py에 있는 이상적인 육각형 좌표를 가져옵니다.
        # (중요) OpenCV 이미지는 Y축이 아래로 갈수록 값이 커지지만(+),
        # 일반적인 수학 그래프는 Y축이 위로 갈수록 커집니다.
        # 그래서 -cfg.TEMPLATE_Y를 해서 Y축을 뒤집어 줍니다.
        self.template_pts = np.column_stack((cfg.TEMPLATE_X, -cfg.TEMPLATE_Y)).astype(np.float32)

        # 2. 캘리브레이션 데이터 로드
        self.mtx = None
        self.dist = None
        self._load_calib_data()

    def _load_calib_data(self):
        """
        [내부 함수] 렌즈 보정값 불러오기
        """
        try:
            # 미리 찍어둔 체커보드 데이터(.npz)를 엽니다.
            with np.load(cfg.CALIB_FILE) as data:
                self.mtx, self.dist = data['mtx'], data['dist']
        except:
            # 파일이 없으면 보정을 못 하므로 그냥 넘어갑니다. (테스트용 안전장치)
            pass 

    def load_and_calibrate(self, img_path):
        """
        [이미지 로드] 파일을 읽고 렌즈 왜곡을 폅니다.
        """
        # 1. 파일 읽기
        img = cv2.imread(img_path)
        if img is None: return None # 파일이 없거나 깨졌으면 종료
        
        # 2. 왜곡 보정 (Undistort)
        # 카메라 렌즈는 둥글기 때문에 사진 가장자리가 휘어 보입니다.
        # 이를 펴주지 않으면 치수 측정 오차가 발생하므로, mtx/dist 값을 이용해 평평하게 폅니다.
        if self.mtx is not None and self.dist is not None:
            return cv2.undistort(img, self.mtx, self.dist, None, None)
        
        return img # 보정 데이터가 없으면 원본 그대로 반환

    def analyze(self, img):
        """
        [1차 분석] 이미지에서 '너트'라고 생각되는 물체를 찾습니다.
        여기서는 '위치'와 '외곽선 점들'을 확보하는 것이 목표입니다.
        """
        # 1. 전처리 (Preprocessing)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 흑백 변환 (데이터 양을 1/3로 줄여 속도 향상)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)  # 블러링 (자잘한 노이즈를 흐리게 뭉갬)
        
        # 2. 이진화 (Thresholding)
        # 이미지를 흰색(배경)과 검은색(물체) 두 가지 색으로만 나눕니다.
        # Otsu 알고리즘을 써서 가장 적절한 경계값을 자동으로 찾습니다.
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 3. 모폴로지 연산 (Morphology)
        # 물체 내부에 생긴 작은 구멍이나 점들을 메워줍니다. (단단한 마스크 생성)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        
        # 4. 가장자리 지우기 (마진)
        # 사진 테두리에 생긴 검은 줄이나 노이즈를 분석에서 제외하기 위해 강제로 지웁니다.
        h, w = mask.shape
        m = cfg.CROP_MARGIN
        mask[:m, :] = 0; mask[h-m:, :] = 0; mask[:, :m] = 0; mask[:, w-m:] = 0
        
        # 5. 외곽선 찾기 (Find Contours)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None # 물체가 없으면 종료

        # 6. 가장 큰 덩어리 선택
        # 잡다한 먼지가 잡힐 수도 있으니, 면적이 가장 큰 것이 너트라고 가정합니다.
        largest_cnt = max(contours, key=cv2.contourArea)
        
        # (옵션) 구멍 메우기: 무게중심을 정확히 구하기 위해 내부를 꽉 채운 마스크를 만듭니다.
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, [largest_cnt], -1, 255, cv2.FILLED)
        
        # 7. 무게중심(Centroid) 계산
        # 모멘트(Moments)라는 수학적 방법을 써서 덩어리의 정중앙 좌표(cx, cy)를 구합니다.
        M = cv2.moments(largest_cnt)
        if M["m00"] == 0: return None
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        
        # 8. 정밀 외곽선 추출 (Approximation)
        # (핵심) EPSILON을 0.0001로 아주 낮게 설정했습니다.
        # 이는 곡선을 단순화하지 않고, 미세한 톱니바퀴 모양까지 다 살려서 점으로 저장하겠다는 뜻입니다.
        peri = cv2.arcLength(largest_cnt, True)
        approx = cv2.approxPolyDP(largest_cnt, cfg.APPROX_EPSILON * peri, True)
        
        return {
            "mask": mask,         # 구멍 찾기용 원본 마스크
            "cnt": largest_cnt,   # 너트 덩어리
            "approx": approx,     # 수천 개의 정밀 좌표 점
            "center": (cx, cy),   # 중심점
            "area": cv2.contourArea(largest_cnt) # 면적
        }

    def find_best_angle(self, data):
        """
        [2차 분석] 삐딱한 너트를 정면(12시)으로 돌리기 위한 '최적 각도'를 찾습니다.
        """
        approx = data['approx']
        cx, cy = data['center']
        
        # 1. 좌표계 변환: (0,0)을 중심으로 이동
        pts = approx.reshape(-1, 2)
        pts_c = pts - [cx, cy]
        pts_c[:, 1] *= -1 # Y축 반전 (이미지 좌표 -> 수학 그래프 좌표)
        
        # 2. 1차 대략적 정렬 (가장 먼 점 기준)
        # 중심에서 가장 먼 점(꼭짓점 중 하나)을 찾아서 그 점이 몇 도에 있는지 계산합니다.
        dists = np.sqrt(pts_c[:,0]**2 + pts_c[:,1]**2)
        idx = np.argmax(dists)
        fx, fy = pts_c[idx]
        # 그 점을 90도(12시)로 보내려면 몇 도를 돌려야 하는지 계산합니다.
        rough_angle = 90.0 - np.degrees(np.arctan2(fy, fx))
        
        # 3. 2차 정밀 정렬 (템플릿 매칭) - [알고리즘의 핵심]
        # "가장 먼 점"이 불량이라서 툭 튀어나와 있다면, 1차 각도는 틀렸을 겁니다.
        # 그래서 그 주변 +/- 3도를 0.1도씩 돌려가며 '모든 점'을 정답 틀에 맞춰봅니다.
        best_angle = rough_angle
        min_error = float('inf')
        
        # 속도를 위해 점을 5개씩 건너뛰며 샘플링(Sampling)해서 검사
        sample_pts = pts_c[::5] 
        
        for angle in np.arange(rough_angle - 3.0, rough_angle + 3.0, 0.1):
            # 회전 행렬 계산
            rad = np.radians(angle)
            cos_v, sin_v = np.cos(rad), np.sin(rad)
            
            current_max = 0
            # 모든 샘플 점들을 이 각도로 회전시켜 봅니다.
            for px, py in sample_pts:
                nx = px * cos_v - py * sin_v
                ny = px * sin_v + py * cos_v
                
                # "이 점이 정답 선에서 얼마나 벗어났니?" (거리 계산)
                dist = abs(cv2.pointPolygonTest(self.template_pts, (float(nx), float(-ny)), True))
                
                # 가장 많이 벗어난 거리(Max Error)를 기록합니다.
                if dist > current_max: current_max = dist
            
            # "가장 많이 벗어난 거리"가 "가장 작은" 각도가 정답입니다. (Minimax)
            # 즉, 모든 점을 정답 틀 안으로 최대한 욱여넣을 수 있는 각도를 찾습니다.
            if current_max < min_error:
                min_error = current_max
                best_angle = angle
                
        return best_angle

    def inspect(self, data, angle):
        """
        [3차 분석] 찾은 각도로 최종 회전시키고, 진짜 불량인지 판정합니다.
        """
        approx = data['approx']
        cx, cy = data['center']
        
        # 1. 최종 회전 적용
        rad = np.radians(angle)
        cos_v, sin_v = np.cos(rad), np.sin(rad)
        
        pts = approx.reshape(-1, 2)
        pts_c = pts - [cx, cy]; pts_c[:, 1] *= -1
        
        rot_x, rot_y = [], []
        for px, py in pts_c:
            rot_x.append(px * cos_v - py * sin_v)
            rot_y.append(px * sin_v + py * cos_v)
            
        # 2. 각도 순서로 데이터 정렬 (P0 -> P1 -> P2...)
        # 이걸 해줘야 그래프 X축(0~360도)과 데이터가 딱 맞아떨어집니다.
        angles = np.degrees(np.arctan2(rot_y, rot_x))
        angles_clk = (90 - angles) % 360
        idx = np.argsort(angles_clk)
        rot_x = np.array(rot_x)[idx]
        rot_y = np.array(rot_y)[idx]
        
        # 3. 형상 불량 판정
        max_dist = 0
        for i in range(len(rot_x)):
            pt = (float(rot_x[i]), float(-rot_y[i]))
            # 정답 템플릿과의 거리 측정 (소수점 단위 정밀도)
            dist = abs(cv2.pointPolygonTest(self.template_pts, pt, True))
            if dist > max_dist: max_dist = dist
            
        # 결과 문자열 결정
        shape_res = "OK"
        if max_dist > cfg.LIMIT_FAIL: shape_res = "FAIL"  # 6.0px 초과
        elif max_dist > cfg.LIMIT_WARNING: shape_res = "WARN" # 4.5 ~ 6.0px
        
        # 4. 구멍(Hole) 검사
        hole_info = {"found": False, "offset": 0, "res": "NO_HOLE", "rot_x":[], "rot_y":[], "rot_center":(0,0)}
        
        # 구멍은 '마스크' 내부의 빈 공간(Contour)을 다시 찾아서 분석합니다.
        cnts_h, _ = cv2.findContours(data['mask'], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 면적이 큰 순서로 정렬 (0번=몸체, 1번=구멍 일 확률 높음)
        cnts_h = sorted(cnts_h, key=cv2.contourArea, reverse=True)
        ppm = cfg.PIXELS_PER_MM
        if len(cnts_h) > 1 and cv2.contourArea(cnts_h[1]) > 100: # 너무 작은 노이즈 제외
            h_cnt = cnts_h[1]
            M = cv2.moments(h_cnt)
            if M["m00"] != 0:
                hx, hy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]) # 구멍 중심
                
                # 편심 거리 계산 (피타고라스 정리)
                offset = np.sqrt((cx-hx)**2 + (cy-hy)**2)
                
                # 구멍 좌표들도 그래프에 그리기 위해 회전시킵니다.
                h_pts = h_cnt.reshape(-1, 2)
                h_rot_x = (h_pts[:,0]-cx)*cos_v - (cy-h_pts[:,1])*sin_v
                h_rot_y = (h_pts[:,0]-cx)*sin_v + (cy-h_pts[:,1])*cos_v
                
                hc_rot_x = (hx-cx)*cos_v - (cy-hy)*sin_v
                hc_rot_y = (hx-cx)*sin_v + (cy-hy)*cos_v
                
                hole_res = "OK" if offset <= cfg.TOL_HOLE else "FAIL"
                
                hole_info = {
                    "found": True, "offset": offset, "offset_mm": offset / ppm, "res": hole_res, "cnt": h_cnt, 
                    "center": (hx, hy), "rot_x": h_rot_x, "rot_y": h_rot_y, "rot_center": (hc_rot_x, hc_rot_y)
                }
        max_dist_mm = max_dist / ppm            # 길이(mm) = 픽셀 / 비율
        area_mm2 = data['area'] / (ppm * ppm)   # 면적(mm^2) = 픽셀 / (비율^2)

        # 모든 분석 결과를 딕셔너리로 묶어서 반환
        return {
            "shape": {"x": rot_x, "y": rot_y, "max_dist": max_dist, "max_dist_mm": max_dist_mm, "res": shape_res},
            "hole": hole_info,
            "angle": angle,
            "area_mm2": area_mm2,           #  mm^2 면적
            "center": data['center']        # 중심점 좌표 (이거왜 누락되어있었음??)
        }