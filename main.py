import os
import json
import asyncio
import cv2
import base64
import time
import numpy as np
from typing import Dict, List
import paho.mqtt.client as mqtt
from functools import partial

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

# --- [모듈 임포트] ---
from models import get_db
from vali import run_inspection
from routers import user_router, control_router, line_router, log_router
from ai_core import AI_Analyzer
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 현재 폴더 경로 추가
from vali.run_inspection import run_algorithm

# --- [설정 및 초기화] ---
app = FastAPI()

app.include_router(user_router.router)
app.include_router(control_router.router)
app.include_router(line_router.router)
app.include_router(log_router.router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "static/images")
TEMP_DIR = os.path.join(BASE_DIR, "static/temp_inspection") # 검사용 임시 폴더
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# 상태 관리 (메모리)
batch_store: Dict[str, dict] = {}

# --- [상태 관리 전역 변수] ---
CURRENT_SHUTTER_STATE = "UP"
LAST_FRAMES: Dict[int, str] = {1: None, 2: None}     # 웹소켓 전송용 (Base64)
LATEST_FRAME_CV: Dict[int, np.ndarray] = {1: None, 2: None} # 검사용 원본 (OpenCV객체)
LAST_SAVE_TIME = {1: 0, 2: 0}
SAVE_INTERVAL = 0.5 
FRAME_COUNTERS: Dict[int, int] = {1: 0, 2: 0}

# MQTT 설정
MQTT_BROKER = "localhost" # 도커 서비스명 (로컬 실행 시 "localhost")
MQTT_PORT = 1883
MQTT_TOPIC_SHUTTER = "factory/shutter/command"  # 기존: 셔터 제어 (UP/DOWN)
MQTT_TOPIC_COMMAND = "factory/command"

# 웹소켓 매니저
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {1: [], 2: []}
    async def connect(self, websocket: WebSocket, camera_index: int):
        await websocket.accept()
        # 해당 카메라 방에 시청자 추가
        if camera_index not in self.active_connections:
            self.active_connections[camera_index] = []
        self.active_connections[camera_index].append(websocket)

    def disconnect(self, websocket: WebSocket, camera_index: int):
        if camera_index in self.active_connections:
            if websocket in self.active_connections[camera_index]:
                self.active_connections[camera_index].remove(websocket)

    # 특정 카메라 방에 있는 사람들에게만 전송
    async def broadcast_bytes(self, data: bytes, camera_index: int):
        if camera_index in self.active_connections:
            for connection in self.active_connections[camera_index]:
                try:
                    await connection.send_bytes(data)
                except:
                    pass

manager = ConnectionManager()

# --- [검사 프로세스 관리자] ---
class InspectionManager:
    def __init__(self):
        self.is_inspecting = False
        self.step = 0 # 0:대기, 1:1차촬영대기, 2:2차촬영대기
        self.cam1_file = ""
        self.cam2_file = ""

    async def start_inspection(self):
        if self.is_inspecting:
            print("⚠️ [Inspect] 이미 검사가 진행 중입니다.")
            return
        
        print("\n🚀 [Inspect] 정밀 검사 프로세스 시작!")
        self.is_inspecting = True
        self.step = 1
        
        print("   -> [Step 1] 셔터 UP 요청")
        send_mqtt("UP")
        # 이제 UP_DONE이 올 때까지 대기

    async def on_up_done(self):
        if not self.is_inspecting or self.step != 1: return

        print("   -> [Step 2] 셔터 닫힘 확인. Camera 1 촬영...")
        await asyncio.sleep(0.5) # 물리적 진동 안정화 대기
        
        # Camera 1 최신 프레임 캡처 및 저장
        if LATEST_FRAME_CV[1] is not None:
            filename = f"ins_cam1_{int(time.time())}.jpg"
            self.cam1_file = os.path.join(TEMP_DIR, filename)
            cv2.imwrite(self.cam1_file, LATEST_FRAME_CV[1])
            print(f"      📸 Cam 1 저장 완료: {filename}")
        else:
            print("      ❌ Cam 1 영상이 없습니다! (검사 실패)")
            self.reset()
            return

        self.step = 2
        print("   -> [Step 3] 셔터 DOWN 요청")
        send_mqtt("DOWN")
        # 이제 DOWN_DONE이 올 때까지 대기

    async def on_down_done(self):
        if not self.is_inspecting or self.step != 2: return

        print("   -> [Step 4] 셔터 열림 확인. Camera 2 촬영...")
        await asyncio.sleep(0.5) 
        
        # Camera 2 최신 프레임 캡처 및 저장
        if LATEST_FRAME_CV[2] is not None:
            filename = f"ins_cam2_{int(time.time())}.jpg"
            self.cam2_file = os.path.join(TEMP_DIR, filename)
            cv2.imwrite(self.cam2_file, LATEST_FRAME_CV[2])
            print(f"      📸 Cam 2 저장 완료: {filename}")
        else:
            print("      ❌ Cam 2 영상이 없습니다! (검사 실패)")
            self.reset()
            return

        # 3. 최종 알고리즘 실행
        print("   -> [Step 5] 검사 알고리즘 실행 (run_algorithm)")
        
        # run_algorithm은 동기 함수이므로 쓰레드로 실행
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_algorithm, self.cam1_file, self.cam2_file)
        
        if result == 1:
            print("✅ [Inspect] 검사 성공 (DB 저장 완료)")
        else:
            print("❌ [Inspect] 검사 실패 (알고리즘 오류)")
            
        self.reset()
        print("   -> [] 셔터 UP 요청")
        send_mqtt("UP")

    def reset(self):
        self.is_inspecting = False
        self.step = 0
        self.cam1_file = ""
        self.cam2_file = ""
        print("⏹ [Inspect] 프로세스 종료 (대기 상태 복귀)\n")

inspection_mgr = InspectionManager()


# --- [Paho MQTT 설정] ---

def send_mqtt(command):
    """ MQTT 메시지 발행 헬퍼 """
    try:
        mqtt_client.publish(MQTT_TOPIC_SHUTTER, command)
    except Exception as e:
        print(f"❌ MQTT 전송 실패: {e}")

def on_connect(client, userdata, flags, rc):
    print(f"📡 [MQTT] 브로커 연결 성공 (Code: {rc})")
    # 명령 토픽 구독
    client.subscribe(MQTT_TOPIC_COMMAND)

def on_message(client, userdata, msg):
    global CURRENT_SHUTTER_STATE
    try:
        topic = msg.topic
        payload = msg.payload.decode().upper()
        print(f"📩 [MQTT] {topic} : {payload}")

        # 비동기 함수 호출을 위한 루프 가져오기
        loop = asyncio.get_event_loop()

        if topic == MQTT_TOPIC_COMMAND:
            if payload == "CHECK":
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(inspection_mgr.start_inspection(), loop)
            
            elif payload == "DOWN_DONE":
                CURRENT_SHUTTER_STATE = "DOWN"
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(inspection_mgr.on_down_done(), loop)
                    
            elif payload == "UP_DONE":
                CURRENT_SHUTTER_STATE = "UP"
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(inspection_mgr.on_up_done(), loop)

    except Exception as e:
        print(f"❌ [MQTT] 에러: {e}")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

@app.on_event("startup")
async def startup_event():
    try:
        mqtt_client.connect(MQTT_BROKER, 1883, 60)
        mqtt_client.loop_start()
    except:
        print("❌ MQTT 연결 실패")

@app.on_event("shutdown")
async def shutdown_event():
    mqtt_client.loop_stop()
# --- [API 엔드포인트] ---




# WebSocket 부분

ai_engine =AI_Analyzer()

@app.websocket("/api/view/{camera_index}")
async def viewer_endpoint(websocket: WebSocket, camera_index: int):
    # 시청자가 들어올 때 "저는 n번 카메라 볼래요"라고 등록
    await manager.connect(websocket, camera_index)
    try:
        while True:
            # 클라이언트(시청자)가 보내는 데이터는 무시 (연결 유지용)
            await websocket.receive()
    except WebSocketDisconnect:
        manager.disconnect(websocket, camera_index)
    except Exception as e:
        print(f"⚠️ [View {camera_index}] 에러: {e}")
        manager.disconnect(websocket, camera_index)


@app.websocket("/ws/source/{camera_index}")
async def source_endpoint(websocket: WebSocket, camera_index: int):
    await websocket.accept()
    print(f"🎥 [Source] 카메라 {camera_index} 송출 시작")
    
    loop = asyncio.get_event_loop()

    try:
        while True:
            # 1. 수신
            data = await websocket.receive_bytes()
            if len(data) == 0: continue
            
            # 2. 디코딩
            nparr = await loop.run_in_executor(None, np.frombuffer, data, np.uint8)
            frame = await loop.run_in_executor(None, cv2.imdecode, nparr, cv2.IMREAD_COLOR)
            
            if frame is None: continue


            LATEST_FRAME_CV[camera_index] = frame

            FRAME_COUNTERS[camera_index] += 1
            final_img = frame 


            if FRAME_COUNTERS[camera_index] % 3 == 0:
                ai_result = await loop.run_in_executor(None, ai_engine.predict, frame)
                if ai_result is not None:
                     if isinstance(ai_result, tuple) and len(ai_result) >= 2:
                        _, predicted_img = ai_result[:2]
                        if predicted_img is not None:
                            final_img = predicted_img
            
            if camera_index == 1 and FRAME_COUNTERS[camera_index] % 5 == 0:
                 await loop.run_in_executor(None, ai_engine.predict, frame)


            _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            

            byte_data = buffer.tobytes()


            LAST_FRAMES[camera_index] = byte_data
            

            await manager.broadcast_bytes(byte_data, camera_index)

    except WebSocketDisconnect:
        print(f"🔌 [Source] 카메라 {camera_index} 연결 끊김")
    except Exception as e:
        print(f"❌ [Source] 에러: {e}")
