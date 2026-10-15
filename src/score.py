"""Weekly scorer: for every completed week not yet scored, write per-student risk scores to HDFS."""
import io
import time

import joblib
import pandas as pd

from common import (MODEL, SCORES, STATIC, exists, flag, completed_weeks, hdfs, load_raw,
                    read_csv, week_dir, write_csv)
from features import snapshot

POLL_SECONDS = 10
CONTEXT = ["att_rate", "sub_rate", "avg_score", "n_failed", "Course"]  # shown on the dashboard


def main():
    client = hdfs()
    while not exists(client, MODEL):
        print("waiting for model...")
        time.sleep(POLL_SECONDS)
    # ponytail: model loaded once; restart the scorer after retraining
    with client.read(MODEL) as f:
        bundle = joblib.load(io.BytesIO(f.read()))
    model, cols = bundle["model"], bundle["features"]
    static = read_csv(client, STATIC).drop(columns="Target")

    raw_by_week = {}  # cache: each week's raw events are read from HDFS once
    while True:
        for week in completed_weeks(client):
            out = f"{week_dir(SCORES, None, week)}/scores.csv"
            if week not in raw_by_week:
                raw_by_week[week] = load_raw(client, [week])
            if exists(client, out):
                continue
            raw = {t: pd.concat([raw_by_week[w][t] for w in range(1, week + 1)]) for t in raw_by_week[week]}
            df = snapshot(raw, week, static["student_id"]).merge(static, on="student_id")
            df["risk_score"] = model.predict_proba(df[cols])[:, 1].round(4)
            df["flagged"] = flag(df["risk_score"])
            write_csv(client, out, df[["student_id", "week", "risk_score", "flagged"] + CONTEXT])
            print(f"week {week:02d}: {df.flagged.sum()} of {len(df)} students flagged -> {out}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
