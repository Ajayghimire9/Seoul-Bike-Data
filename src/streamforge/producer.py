import json

from confluent_kafka import Producer


def publish_rows(
    rows: list[dict], bootstrap: str = "localhost:9092", topic: str = "bike-demand"
) -> int:
    producer = Producer({"bootstrap.servers": bootstrap})
    for row in rows:
        producer.produce(topic, value=json.dumps(row, default=str).encode("utf-8"))
    producer.flush()
    return len(rows)
