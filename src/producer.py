"""Simulate weekly attendance / assignment / assessment events and publish them to Kafka.

The UCI dataset only has semester totals, so each student's totals are spread over
16 weeks per semester. A fixed seed makes every run identical.
"""
import json
import os
import time

import numpy as np
from kafka import KafkaProducer

from common import KAFKA_BOOTSTRAP, SEED, TOPICS, WEEKS, WEEKS_PER_SEM, completed_weeks, hdfs
from download import load_local

DEFAULT_LOAD = 5  # sessions/week scheduled for a student enrolled in no units


def simulate(df, seed=SEED):
    """Return {week: {topic: [event, ...]}} for weeks 1..WEEKS."""
    rng = np.random.default_rng(seed)
    out = {w: {t: [] for t in TOPICS} for w in range(1, WEEKS + 1)}
    ids = df["student_id"].to_numpy()
    for sem, name in ((1, "1st"), (2, "2nd")):
        col = lambda k: df[f"Curricular units {name} sem ({k})"].to_numpy()  # noqa: E731
        enrolled, evals, approved, grade, no_eval = (
            col("enrolled"), col("evaluations"), col("approved"), col("grade"), col("without evaluations"))
        active = enrolled > 0
        ratio = np.where(active, approved / np.maximum(enrolled, 1), 0.0)
        skipped = no_eval / np.maximum(enrolled, 1)
        sessions = np.where(active, enrolled, DEFAULT_LOAD)
        att_noise = rng.normal(0, 0.05, len(df))
        sub_noise = rng.normal(0, 0.05, len(df))

        for k in range(1, WEEKS_PER_SEM + 1):
            week = (sem - 1) * WEEKS_PER_SEM + k
            decline = (1 - ratio) * k / WEEKS_PER_SEM  # struggling students fade over the semester
            p_att = np.where(active, np.clip(0.55 + 0.4 * ratio - 0.3 * skipped - 0.25 * decline + att_noise, 0.02, 0.99), 0.03)
            p_sub = np.where(active, np.clip(0.35 + 0.6 * ratio - 0.2 * decline + sub_noise, 0.02, 0.99), 0.02)
            attended = rng.binomial(sessions, p_att)
            submitted = rng.binomial(sessions, p_sub)
            for i in range(len(df)):
                sid = int(ids[i])
                out[week]["attendance"].append(
                    {"student_id": sid, "week": week, "sessions": int(sessions[i]), "attended": int(attended[i])})
                out[week]["assignments"].append(
                    {"student_id": sid, "week": week, "assigned": int(sessions[i]), "submitted": int(submitted[i])})

        # assessments: `evaluations` exams at random weeks, `approved` of them passed
        for i in range(len(df)):
            n = int(evals[i])
            if n == 0:
                continue
            n_pass = min(int(approved[i]), n)
            passed = rng.permutation([1] * n_pass + [0] * (n - n_pass))
            weeks = np.sort(rng.integers(1, WEEKS_PER_SEM + 1, n)) + (sem - 1) * WEEKS_PER_SEM
            for w, p in zip(weeks, passed):
                score = np.clip(rng.normal(grade[i], 1.5), 10, 20) if p else rng.uniform(2, 9.5)
                out[int(w)]["assessments"].append(
                    {"student_id": int(ids[i]), "week": int(w), "score": round(float(score), 1), "passed": int(p)})
    return out


if __name__ == "__main__":
    if len(completed_weeks(hdfs())) == WEEKS:
        print("all weeks already in HDFS; nothing to publish (docker compose down -v to start over)")
        raise SystemExit
    delay = float(os.getenv("WEEK_DELAY_SECONDS", "3"))
    events = simulate(load_local())
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        key_serializer=lambda k: k and str(k).encode(),
        value_serializer=lambda v: json.dumps(v).encode(),
        linger_ms=50,
        acks="all",
    )
    for week in range(1, WEEKS + 1):
        for topic in TOPICS:
            for ev in events[week][topic]:
                producer.send(topic, key=ev["student_id"], value=ev)
            producer.send(topic, value={"type": "week_end", "week": week})
        producer.flush()
        print(f"week {week:02d}: " + ", ".join(f"{t}={len(events[week][t])}" for t in TOPICS))
        time.sleep(delay)
    producer.close()
