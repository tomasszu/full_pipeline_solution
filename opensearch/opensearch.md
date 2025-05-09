# Docker Opensearch datubāzes instance

### Container izveide

```bash
docker pull opensearchproject/opensearch:latest && docker run -it -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" opensearchproject/opensearch:latest

```

#### Tip!
(Ja kkadi memory errori tad var meginat docker prune vai vsp padzest liekos failus aaraa)

Vai additionally, samazina atminas warningus:

```bash
docker exec -it <container_id> curl -XPUT "localhost:9200/_cluster/settings" -H "Content-Type: application/json" -d '{
  "persistent": {
    "cluster.routing.allocation.disk.watermark.low": "5gb",
    "cluster.routing.allocation.disk.watermark.high": "3gb",
    "cluster.routing.allocation.disk.watermark.flood_stage": "1gb",
    "cluster.info.update.interval": "1m"
  }
}'

```

### Izveido vektor index

create_vector_index.json


### Verify, ka container ir running:

```bash
curl -X GET http://localhost:9200

```

### Help tools

delete_index.json

add_data_to_index.json

search_data.json

