import paho.mqtt.client as mqtt
import json
import time

MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "tomass/detections"

client = mqtt.Client()

# Optional: Define callbacks to debug
def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))  # 0 means OK

def on_publish(client, userdata, mid):
    print("Message published, mid=", mid)

client.on_connect = on_connect
client.on_publish = on_publish

client.connect(MQTT_BROKER, MQTT_PORT)
client.loop_start()

# Dummy payload
payload = {
    "track_id": 1,
    "bbox": [100, 150, 200, 250],
    "encoded_crop": "TESTSTRING"
}

# Publish
result = client.publish(MQTT_TOPIC, json.dumps(payload))
print("Publish result:", result)

time.sleep(2)  # Give time for send to complete

client.loop_stop()
client.disconnect()
