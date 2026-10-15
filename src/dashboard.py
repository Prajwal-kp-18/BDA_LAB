"""Advisor dashboard: weekly dropout-risk scores, flagged students, per-student trend."""
import json

import altair as alt
import pandas as pd
import streamlit as st

from common import METRICS, RISK_THRESHOLD, SCORES, exists, hdfs, read_csv, read_text

SERIES = "#2a78d6"     # categorical slot 1 (validated reference palette)
CRITICAL = "#d03b3b"   # status: critical, always paired with a text label
COURSES = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design",
    8014: "Social Service (evening)", 9003: "Agronomy", 9070: "Communication Design",
    9085: "Veterinary Nursing", 9119: "Informatics Engineering", 9130: "Equinculture",
    9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management",
    9773: "Journalism and Communication", 9853: "Basic Education", 9991: "Management (evening)",
}

st.set_page_config(page_title="Dropout Risk", page_icon="🎓", layout="wide")


@st.cache_resource
def client():
    return hdfs()


@st.cache_data(ttl=30)
def load_scores():
    c = client()
    if not exists(c, SCORES):
        return pd.DataFrame()
    frames = []
    for d in sorted(c.list(SCORES)):
        try:
            frames.append(read_csv(c, f"{SCORES}/{d}/scores.csv"))
        except Exception:  # week being written right now; picked up on next refresh
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["course"] = df["Course"].map(COURSES).fillna(df["Course"].astype(str))
    return df


@st.cache_data(ttl=300)
def load_metrics():
    c = client()
    return json.loads(read_text(c, METRICS)) if exists(c, METRICS) else None


def threshold_rule():
    rule = alt.Chart(pd.DataFrame({"t": [RISK_THRESHOLD]}))
    return (rule.mark_rule(color=CRITICAL, strokeDash=[4, 4], strokeWidth=2).encode(y="t:Q")
            + rule.mark_text(text=f"follow-up threshold {RISK_THRESHOLD}", align="left", dx=4, dy=-6,
                             color=CRITICAL).encode(y="t:Q", x=alt.value(0)))


st.title("Student dropout risk: advisor view")
if st.button("Refresh data"):
    st.cache_data.clear()

scores = load_scores()
if scores.empty:
    st.info("No scores yet. The pipeline is still ingesting data or training the model; refresh in a minute.")
    st.stop()

weeks = sorted(scores["week"].unique())
with st.sidebar:
    week = st.selectbox("Week", weeks, index=len(weeks) - 1)
    courses = st.multiselect("Course", sorted(scores["course"].unique()), placeholder="All courses")

view = scores if not courses else scores[scores["course"].isin(courses)]
now = view[view["week"] == week]
prev = view[view["week"] == week - 1]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Students scored", f"{len(now):,}")
k2.metric(f"Flagged (risk > {RISK_THRESHOLD})", f"{now.flagged.sum():,}",
          delta=int(now.flagged.sum() - prev.flagged.sum()) if len(prev) else None, delta_color="inverse")
k3.metric("Share flagged", f"{now.flagged.mean():.1%}")
k4.metric("Median risk", f"{now.risk_score.median():.2f}")

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Flagged students: week {week}")
    flagged = now[now.flagged].sort_values("risk_score", ascending=False)
    table = flagged.assign(att_rate=flagged.att_rate * 100, sub_rate=flagged.sub_rate * 100)[
        ["student_id", "course", "risk_score", "att_rate", "sub_rate", "avg_score", "n_failed"]]
    st.dataframe(table, hide_index=True, width="stretch", height=420, column_config={
        "student_id": "Student",
        "course": "Course",
        "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=1, format="%.2f"),
        "att_rate": st.column_config.NumberColumn("Attendance %", format="%.0f"),
        "sub_rate": st.column_config.NumberColumn("Submitted %", format="%.0f"),
        "avg_score": st.column_config.NumberColumn("Avg score /20", format="%.1f"),
        "n_failed": st.column_config.NumberColumn("Failed assessments", format="%d"),
    })
    st.download_button("Download flagged list (CSV)", flagged.to_csv(index=False),
                       file_name=f"flagged_week_{week:02d}.csv", mime="text/csv")
with right:
    st.subheader("Risk distribution")
    hist = alt.Chart(now).mark_bar(color=SERIES, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        y=alt.Y("risk_score:Q", bin=alt.Bin(step=0.05), title="Risk score"),
        x=alt.X("count():Q", title="Students"),
        tooltip=[alt.Tooltip("risk_score:Q", bin=alt.Bin(step=0.05), title="Risk"),
                 alt.Tooltip("count():Q", title="Students")],
    )
    st.altair_chart(hist + threshold_rule(), use_container_width=True)

st.subheader("Student drill-down")
options = now.sort_values("risk_score", ascending=False)["student_id"].tolist()
sid = st.selectbox("Student (highest risk first)", options)
history = view[view["student_id"] == sid].sort_values("week")
last = history[history["week"] == week].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Risk this week", f"{last.risk_score:.2f}", "flagged" if last.flagged else "not flagged",
          delta_color="off")
c2.metric("Attendance", f"{last.att_rate:.0%}")
c3.metric("Assignments submitted", f"{last.sub_rate:.0%}")
c4.metric("Failed assessments", int(last.n_failed))
trend = alt.Chart(history).mark_line(color=SERIES, strokeWidth=2, point=alt.OverlayMarkDef(size=64)).encode(
    x=alt.X("week:Q", title="Week", scale=alt.Scale(domain=[1, max(weeks)])),
    y=alt.Y("risk_score:Q", title="Risk score", scale=alt.Scale(domain=[0, 1])),
    tooltip=[alt.Tooltip("week:Q", title="Week"), alt.Tooltip("risk_score:Q", title="Risk", format=".2f"),
             alt.Tooltip("att_rate:Q", title="Attendance", format=".0%"),
             alt.Tooltip("n_failed:Q", title="Failed assessments")],
)
st.altair_chart(trend + threshold_rule(), use_container_width=True)
with st.expander("Data table"):
    st.dataframe(history, hide_index=True, width="stretch")

metrics = load_metrics()
if metrics:
    with st.expander("Model evaluation (held-out students)"):
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m2.metric(f"Precision @ {RISK_THRESHOLD}", f"{metrics['precision']:.3f}")
        m3.metric(f"Recall @ {RISK_THRESHOLD}", f"{metrics['recall']:.3f}")
        m4.metric("F1", f"{metrics['f1']:.3f}")
        st.write("Share of eventual dropouts already flagged by week:",
                 {f"week {w}": f"{v:.0%}" for w, v in metrics["early_recall"].items()})
