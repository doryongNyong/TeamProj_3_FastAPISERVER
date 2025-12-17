from fastapi import APIRouter, HTTPException
import paho.mqtt.client as mqtt

router = APIRouter(
    prefix="/api",
    tags=["Control"]
)

# MQTT 설정
MQTT_BROKER = "localhost" # docker-compose 서비스명
MQTT_PORT = 1883
MQTT_TOPIC = "/factory/control" # 요청하신 토픽

def send_mqtt_message(topic: str, message: str):
    """ MQTT 브로커로 메시지를 전송하는 헬퍼 함수 """
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

@router.post("/down")
def trigger_down():
    """
    /api/down 호출 시 -> MQTT '/factory/control' 토픽에 'DOWN' 전송
    """
    success = send_mqtt_message(MQTT_TOPIC, "DOWN")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
        
    return {
        "status": "success",
        "topic": MQTT_TOPIC,
        "message": "DOWN sent"
    }

@router.post("/up")
def trigger_down():
    """
    /api/down 호출 시 -> MQTT '/factory/control' 토픽에 'DOWN' 전송
    """
    success = send_mqtt_message(MQTT_TOPIC, "UP")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
        
    return {
        "status": "success",
        "topic": MQTT_TOPIC,
        "message": "UP sent"
    }

@router.post("/normal")
def trigger_down():
    """
    /api/down 호출 시 -> MQTT '/factory/control' 토픽에 'DOWN' 전송
    """
    success = send_mqtt_message(MQTT_TOPIC, "NORMAL")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
        
    return {
        "status": "success",
        "topic": MQTT_TOPIC,
        "message": "NORMAL sent"
    }

@router.post("/defect")
def trigger_down():
    """
    /api/down 호출 시 -> MQTT '/factory/control' 토픽에 'DOWN' 전송
    """
    success = send_mqtt_message(MQTT_TOPIC, "DEFECT")
    
    if not success:
        raise HTTPException(status_code=500, detail="MQTT 전송 실패")
        
    return {
        "status": "success",
        "topic": MQTT_TOPIC,
        "message": "DEFECT sent"
    }