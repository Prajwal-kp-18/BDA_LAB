"""Train the dropout classifier on HDFS data and save model + metrics back to HDFS.

--wait  block until all weeks have landed in HDFS (used by the compose `trainer` service).
"""
import io
import json
import sys
import time

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from common import (METRICS, MODEL, RISK_THRESHOLD, SEED, STATIC, WEEKS, completed_weeks,
                    exists, flag, hdfs, load_raw, read_csv, put)
from features import build, feature_columns

EARLY_WEEKS = [4, 8, 12]


def main(wait):
    client = hdfs()
    while wait and (len(completed_weeks(client)) < WEEKS or not exists(client, STATIC)):
        print(f"waiting for data: {len(completed_weeks(client))}/{WEEKS} weeks in HDFS")
        time.sleep(10)
    weeks = completed_weeks(client)
    if not weeks:
        sys.exit("no completed weeks in HDFS yet")

    static = read_csv(client, STATIC)
    df = build(load_raw(client, weeks), weeks, static)
    df["y"] = df["student_id"].map(static.set_index("student_id")["Target"].eq("Dropout").astype(int))
    cols = feature_columns(df.drop(columns="y"))

    # split by student so no student is in both train and test
    labels = static.set_index("student_id")["Target"].eq("Dropout")
    train_ids, test_ids = train_test_split(labels.index, test_size=0.2, stratify=labels, random_state=SEED)
    train, test = df[df.student_id.isin(train_ids)], df[df.student_id.isin(test_ids)]

    model = HistGradientBoostingClassifier(random_state=SEED).fit(train[cols], train["y"])
    test = test.assign(risk=model.predict_proba(test[cols])[:, 1])
    pred = flag(test["risk"])

    dropouts = test[test.y == 1]
    metrics = {
        "weeks_trained": len(weeks),
        "train_students": len(train_ids),
        "test_students": len(test_ids),
        "threshold": RISK_THRESHOLD,
        "roc_auc": roc_auc_score(test["y"], test["risk"]),
        "precision": precision_score(test["y"], pred),
        "recall": recall_score(test["y"], pred),
        "f1": f1_score(test["y"], pred),
        # share of true dropouts already flagged at week N
        "early_recall": {w: float(flag(dropouts[dropouts.week == w]["risk"]).mean())
                         for w in EARLY_WEEKS if w in weeks},
        "auc_by_week": {int(w): roc_auc_score(g["y"], g["risk"]) for w, g in test.groupby("week")},
    }

    buf = io.BytesIO()
    joblib.dump({"model": model, "features": cols}, buf)
    put(client, MODEL, buf.getvalue())
    put(client, METRICS, json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k != "auc_by_week"}, indent=2))
    print(f"model -> {MODEL}, metrics -> {METRICS}")


if __name__ == "__main__":
    main("--wait" in sys.argv)
