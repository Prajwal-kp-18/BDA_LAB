"""Shared constants and HDFS helpers for the dropout-risk pipeline."""
import io
import json
import os
import time

import pandas as pd

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
HDFS_URL = os.getenv("HDFS_URL", "http://localhost:9870")

# event fields per topic (every event also carries student_id and week)
SCHEMA = {
    "attendance": ["sessions", "attended"],
    "assignments": ["assigned", "submitted"],
    "assessments": ["score", "passed"],
}
TOPICS = list(SCHEMA)
WEEKS_PER_SEM = 16
WEEKS = 2 * WEEKS_PER_SEM
RISK_THRESHOLD = 0.6
SEED = 42

RAW = "/data/raw"
STATIC = "/data/static/students.csv"
SCORES = "/data/scores"
MODEL = "/models/dropout_model.joblib"
METRICS = "/models/metrics.json"


def flag(risk):
    """Advisor follow-up rule: strictly above the threshold."""
    return risk > RISK_THRESHOLD


def week_dir(root, topic_or_none, week):
    parts = [root] + ([topic_or_none] if topic_or_none else []) + [f"week={week:02d}"]
    return "/".join(parts)


def hdfs():
    from hdfs import InsecureClient
    client = InsecureClient(HDFS_URL, user="root")
    for _ in range(60):  # namenode may still be in safe mode / starting
        try:
            client.status("/")
            return client
        except Exception as e:
            print(f"waiting for HDFS: {e}")
            time.sleep(5)
    raise RuntimeError("HDFS not reachable")


def exists(client, path):
    return client.status(path, strict=False) is not None


def put(client, path, data):
    """Replace an HDFS file (str or bytes) so readers never see it half-written."""
    tmp = f"{path}._tmp"
    client.write(tmp, data=data, overwrite=True, encoding="utf-8" if isinstance(data, str) else None)
    # ponytail: WebHDFS rename can't overwrite, so the path is missing for a few ms
    # between delete and rename; readers see "not found" instead of partial data
    client.delete(path)
    client.rename(tmp, path)


def read_text(client, path):
    with client.read(path, encoding="utf-8") as f:
        return f.read()


def write_csv(client, path, df):
    put(client, path, df.to_csv(index=False))


def read_csv(client, path):
    return pd.read_csv(io.StringIO(read_text(client, path)))


def read_jsonl(client, path):
    return [json.loads(line) for line in read_text(client, path).splitlines() if line]


def completed_weeks(client):
    """Weeks for which every topic has landed in HDFS (all _SUCCESS markers present)."""
    done = set(range(1, WEEKS + 1))
    for topic in TOPICS:
        root = f"{RAW}/{topic}"
        weeks = set()
        if exists(client, root):
            for d in client.list(root):
                if d.startswith("week=") and exists(client, f"{root}/{d}/_SUCCESS"):
                    weeks.add(int(d[5:]))
        done &= weeks
    return sorted(done)


def load_raw(client, weeks):
    """Return {topic: DataFrame} with all events for the given weeks."""
    out = {}
    for topic in TOPICS:
        rows = []
        for w in weeks:
            rows += read_jsonl(client, f"{week_dir(RAW, topic, w)}/part-0.jsonl")
        out[topic] = pd.DataFrame(rows, columns=["student_id", "week"] + SCHEMA[topic])
    return out
