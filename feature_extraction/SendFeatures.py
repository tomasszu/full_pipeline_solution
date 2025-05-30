import paho.mqtt.client as mqtt
import json
import time



class SendFeatures:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1884, mqtt_topic="tomass/save_features"):
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.connected = False

        # Setup MQTT client
        self.client = mqtt.Client(client_id="sender2")
        self.client.on_connect = self.on_connect
        self.client.on_publish = self.on_publish

        self.client.connect(self.mqtt_broker, self.mqtt_port)
        self.client.loop_start()  # Start in background

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected successfully")
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def on_publish(self, client, userdata, mid):
        print("[MQTT] Message published, mid =", mid)
        
    def __call__(self, track_id, features):

        features = features.flatten().tolist()
        
        data = {
            'track_id': int(track_id),
            'features': features  # Convert ndarray to list
        }

        self.send_over_mqtt(data)

    def send_over_mqtt(self, data):

        result = self.client.publish(self.mqtt_topic, json.dumps(data))

        # !! DEBUG

        print(f"[MQTT] sent message for track_id {data['track_id']}")

        if result[0] != 0:
            print(f"[MQTT] Failed to send message for track_id {data['track_id']}")
        
        time.sleep(0.05)  # Let messages flush
