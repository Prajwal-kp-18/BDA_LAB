"""Raw weekly events -> one feature row per (student, week), using only data up to that week."""
import numpy as np
import pandas as pd

TREND_WEEKS = 3
EVENT_FEATURES = [
    "week", "att_rate", "att_trend", "sub_rate", "sub_trend",
    "n_assessments", "avg_score", "n_failed", "fail_rate",
]


def _rate(df, num, den):
    g = df.groupby("student_id")[[num, den]].sum()
    return g[num] / g[den].replace(0, np.nan)


def snapshot(raw, week, student_ids):
    """Features for every student as of the end of `week`."""
    att, asg, ass = (raw[t] for t in ("attendance", "assignments", "assessments"))
    upto = lambda df: df[df["week"] <= week]  # noqa: E731
    recent = lambda df: df[(df["week"] <= week) & (df["week"] > week - TREND_WEEKS)]  # noqa: E731

    f = pd.DataFrame(index=pd.Index(student_ids, name="student_id"))
    f["week"] = week
    f["att_rate"] = _rate(upto(att), "attended", "sessions")
    f["att_trend"] = _rate(recent(att), "attended", "sessions") - f["att_rate"]
    f["sub_rate"] = _rate(upto(asg), "submitted", "assigned")
    f["sub_trend"] = _rate(recent(asg), "submitted", "assigned") - f["sub_rate"]
    g = upto(ass).groupby("student_id")
    f["n_assessments"] = g.size()
    f["avg_score"] = g["score"].mean()  # NaN before the first assessment; the model handles NaN
    f["n_failed"] = g["passed"].apply(lambda p: (p == 0).sum())
    f[["n_assessments", "n_failed"]] = f[["n_assessments", "n_failed"]].fillna(0)
    f["fail_rate"] = f["n_failed"] / f["n_assessments"].replace(0, np.nan)
    return f.reset_index()


def build(raw, weeks, static):
    """Stack snapshots for `weeks` and join the static profile (minus the label)."""
    ids = static["student_id"]
    snaps = pd.concat([snapshot(raw, w, ids) for w in weeks], ignore_index=True)
    return snaps.merge(static.drop(columns=["Target"], errors="ignore"), on="student_id")


def feature_columns(df):
    return [c for c in df.columns if c not in ("student_id", "Target")]
