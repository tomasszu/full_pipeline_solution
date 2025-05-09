import paho.mqtt.client as mqtt
import json
import time


class ReceiveFeatures:
    def __init__(self, broker="localhost", port=1884, topic="tomass/save_features"):

        self.topic = topic
        self.queue = []  # stores (track_id, bbox, timestamp, image_np)
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.connected = False


        self.client.connect(broker, port, 60)
        self.client.loop_start()  # non-blocking

        self.queue = []  # stores (track_id, features)

        # Wait until connection is established
        timeout = time.time() + 5  # max 5 seconds wait
        while not self.connected:
            if time.time() > timeout:
                raise TimeoutError("MQTT connection timed out.")
            time.sleep(0.5)
        

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("[MQTT] Connected successfully")
            client.subscribe(self.topic)  # ← THIS WAS MISSING
            self.connected = True
        else:
            print(f"[MQTT] Connection failed with code {rc}")

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            self.queue.append(payload)

            print(f"[MQTT] Received {payload['track_id']}")
            return payload
        except Exception as e:
            print(f"[ERROR] Failed to process message: {e}")
            return None
        
    def get_pending_vectors(self):
        """Retrieve and clear the queue of received crops"""
        data = self.queue[:]
        self.queue.clear()
        return data




    