# Student Performance & Dropout Risk Prediction (PS 57)

This pipeline streams weekly attendance, assignment and assessment events through **Kafka** and stores them on **HDFS**. A **scikit-learn** classifier then gives every student a dropout-risk score each week. Students scoring **above 0.6** are flagged for an advisor in a **Streamlit** dashboard.

Dataset: UCI [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697) (4,424 students).

## Run

```bash
docker compose up -d          # whole pipeline: ingest -> HDFS -> train -> weekly scoring -> dashboard
open http://localhost:8501    # advisor dashboard
open http://localhost:9870    # HDFS namenode UI
docker compose down -v        # stop and wipe HDFS/Kafka data
```

The first run takes about 5 minutes. The producer publishes one simulated week every `WEEK_DELAY_SECONDS` (default 3). The trainer waits until all 32 weeks are in HDFS, and the scorer then writes one score file per week. If you run `up` again on existing data, the producer skips publishing and the scorer doesn't redo finished weeks.

Useful commands:

```bash
docker compose logs -f producer consumer scorer
docker compose exec namenode hdfs dfs -ls -R /data
docker compose run --rm app python src/train.py      # retrain (then: docker compose restart scorer)
docker compose run --rm app python tests/test_features.py
```

## Architecture

```
data.csv ─► producer.py ─► Kafka topics: attendance | assignments | assessments
                                   │   (+ week_end marker per topic per week)
                                   ▼
                             consumer.py ─► HDFS /data/raw/<topic>/week=NN/part-0.jsonl + _SUCCESS
download.py ─► HDFS /data/static/students.csv (profile + label)
                                   │
              train.py ◄── features.py ──► score.py (polls for completed weeks)
                 │                              │
                 ▼                              ▼
 HDFS /models/dropout_model.joblib    HDFS /data/scores/week=NN/scores.csv
      /models/metrics.json                      │
                                                ▼
                                   dashboard.py (Streamlit :8501)
```

| File | Role |
|---|---|
| `src/common.py` | Topics, paths, `RISK_THRESHOLD = 0.6`, HDFS helpers (atomic write-then-rename) |
| `src/download.py` | Fetches the dataset, adds `student_id`, uploads the static profile |
| `src/producer.py` | Turns semester totals into 32 weeks of events (seed 42) and publishes them |
| `src/consumer.py` | Kafka → HDFS sink; commits offsets only after a week is safely written |
| `src/features.py` | Per-(student, week) features using only data up to that week |
| `src/train.py` | Trains a `HistGradientBoostingClassifier` and evaluates it on held-out students |
| `src/score.py` | Writes `student_id, week, risk_score, flagged` for each new week |
| `src/dashboard.py` | Advisor view: KPIs, flagged list + CSV export, risk distribution, per-student trend |

**Features** (cumulative to week *w*):
- Attendance rate, and its 3-week trend.
- Assignment submission rate, and its 3-week trend.
- Number of assessments, average score, number failed and fail rate.
- The week number.
- Static profile: admission grade, tuition fees up to date, debtor, scholarship, age, course, parents' background, macroeconomic indicators.

The semester outcome columns (`Curricular units …`) are left out of the features because they would leak the label.

## Results

The model was evaluated on 885 held-out students (20%, stratified, split by student), across all 32 weekly snapshots:

| Metric | Value |
|---|---|
| ROC-AUC | 0.911 |
| Precision @ 0.6 | 0.877 |
| Recall @ 0.6 | 0.679 |
| F1 @ 0.6 | 0.765 |
| Share of eventual dropouts flagged by week 4 / 8 / 12 | 66.9% / 67.3% / 67.6% |

AUC rises from about 0.88 at week 1 to about 0.93 at week 32. On the final week, 1,248 of 4,424 students are flagged. The runs are deterministic: a rebuild from scratch reproduces these numbers exactly.

## Limitations

- **The weekly data is simulated.** The UCI dataset only has semester totals. `producer.py` spreads them over 16 weeks per semester: assessments land in random weeks, and the pass/fail split matches the student's approved units.
- **The attendance data is synthetic.** The dataset's "Daytime/evening attendance" column is the class schedule, not presence. Attendance is generated from the student's pass ratio and unevaluated units, plus noise. In the simulation, struggling students drift down over the semester.
- **Most early signal comes from static fields.** Unpaid tuition, debtor status and scholarship already flag about 2/3 of dropouts in week 1. The weekly events add a few AUC points on top.
- **80% of the students on the dashboard were in the training set**, so their scores are in-sample. The metrics above come only from held-out students.
- **The scorer loads the model once.** Restart it after retraining.
