from fastapi import APIRouter, HTTPException
import paho.mqtt.client as mqtt
from pydantic import BaseModel

router = APIRouter(
    prefix="/api",
    tags=["Line Control"]
)

# --- [MQTT 설정] ---
# main.py와 동일한 브로커 정보 사용
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "/factory/control"  # 라인 제어용 토픽

def send_mqtt_command(topic: str, message: str):
    try:
        # 1회성 연결로 메시지 전송 (Publish)
        client = mqtt.Client(client_id="fastapi_control_pub")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.publish(topic, message)
        client.disconnect()
        return True
    except Exception as e:
        print(f"❌ [Control Router] MQTT 전송 실패: {e}")
        return False

# --- [API 엔드포인트] ---

@router.post("/start")
def start_line():
    success = send_mqtt_command(MQTT_TOPIC,"START")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패 (브로커 연결 확인 필요)")
    
    return {
        "status": "success",
        "action": MQTT_TOPIC,
        "message": "공장 라인 가동을 시작합니다."
    }

@router.post("/restart")
def restart_line():
    """ 공장 라인 재가동 """
    success = send_mqtt_command(MQTT_TOPIC,"RESTART")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
    
    return {
        "status": "success",
        "action": MQTT_TOPIC,
        "message": "공장 라인을 재가동합니다."
    }

@router.post("/stop")
def stop_line():
    """ 공장 라인 정지 """
    success = send_mqtt_command(MQTT_TOPIC,"STOP")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
    
    return {
        "status": "success",
        "action": MQTT_TOPIC,
        "message": "공장 라인을 정지합니다."
    }