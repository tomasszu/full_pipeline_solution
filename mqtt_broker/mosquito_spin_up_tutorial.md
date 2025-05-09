# MQTT local instance ReadMe

### Create mosquitto.conf file:

```conf

listener 1884

allow_anonymous true

```


### Spin up container:


verbose:

```bash
docker run -it --rm --name mosquitto_local \
  -v $(pwd)/mosquitto.conf:/mosquitto/config/mosquitto.conf \
  -v $(pwd)/mosquitto_data:/mosquitto/data \
  -p 1884:1884 \
  eclipse-mosquitto

```

or background:

```bash
docker run --name mosquitto_local \
  -v $(pwd)/mosquitto.conf:/mosquitto/config/mosquitto.conf \
  -v $(pwd)/mosquitto_data:/mosquitto/data \
  -p 1884:1884 \
  eclipse-mosquitto

```

### Test Sub and Pub:

```bash
mosquitto_sub -h localhost -p 1884 -t "test/topic"

```

```bash
mosquitto_pub -h localhost -p 1884 -t "test/topic" -m "Message OK!" -r

```