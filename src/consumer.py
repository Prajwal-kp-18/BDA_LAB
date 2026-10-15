"""Kafka -> HDFS sink: buffer each (topic, week), write it as JSONL on week_end, then commit."""
import json
from collections import defaultdict

from kafka import KafkaConsumer, TopicPartition
from kafka.structs import OffsetAndMetadata

from common import KAFKA_BOOTSTRAP, RAW, TOPICS, hdfs, week_dir, put


def main():
    client = hdfs()
    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id="hdfs-sink",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        value_deserializer=lambda b: json.loads(b),
    )
    buffers = defaultdict(list)  # (topic, week) -> [json line]
    for msg in consumer:
        ev = msg.value
        if ev.get("type") != "week_end":
            buffers[(msg.topic, ev["week"])].append(json.dumps(ev))
            continue
        key = (msg.topic, ev["week"])
        lines = buffers.pop(key, [])
        d = week_dir(RAW, msg.topic, ev["week"])
        # overwrite, so a replayed week rewrites the same file instead of duplicating it
        put(client, f"{d}/part-0.jsonl", "\n".join(lines) + "\n" if lines else "")
        put(client, f"{d}/_SUCCESS", "")
        # commit only this topic, only up to this marker: other topics' buffered weeks
        # are not in HDFS yet and must be replayed after a crash
        consumer.commit({TopicPartition(msg.topic, msg.partition): OffsetAndMetadata(msg.offset + 1, "", -1)})
        print(f"{msg.topic} week {ev['week']:02d}: {len(lines)} events -> {d}")


if __name__ == "__main__":
    main()
