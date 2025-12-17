import cv2
from ultralytics import YOLO
from vali import config as cfg
import os

class AIInspector:
    def __init__(self):
        self.model = None
        if os.path.exists(cfg.AI_MODEL_PATH):
            try:
                self.model = YOLO(cfg.AI_MODEL_PATH)
            except: pass

    def inspect(self, img, position_name=""):
        result_safe = {"found": False, "res": "Error", "conf": 0.0, "boxes": []}
        if self.model is None or img is None: return result_safe
        
        try:
            results = self.model.predict(source=img, conf=cfg.AI_CONF_THRES, verbose=False)
            result = results[0]
        except: return result_safe
        
        rust_found = False
        max_conf = 0.0
        boxes_list = []

        if len(result.boxes) > 0:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label_name = self.model.names[cls_id]
                
                if "rust" in label_name.lower():
                    rust_found = True
                    conf = float(box.conf[0])
                    if conf > max_conf: max_conf = conf
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # (!!!) [수정] 좌표 4개 뒤에 점수(conf)를 추가했습니다.
                    boxes_list.append([x1, y1, x2, y2, conf])

        ai_res = "NG" if rust_found else "OK"
        
        if rust_found:
            print(f"   ⚠️ [{position_name}] 녹(Rust) 발견! (확신도: {max_conf:.2f})")
            
        return {
            "found": rust_found,
            "res": ai_res,
            "conf": max_conf,
            "boxes": boxes_list # 이제 여기엔 [x1, y1, x2, y2, conf] 가 들어있습니다.
        }