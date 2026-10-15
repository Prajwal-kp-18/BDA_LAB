# PS 57 — Student Performance & Dropout Risk Prediction

## Context

**Problem statement:** Academic institutions identify at-risk students only after they have already failed multiple assessments, missing the window for early intervention.

**Objective:** Ingest attendance, assignment, and assessment data via **Kafka**, store on **HDFS**, and train a **classification model** to output a **dropout-risk score per student each week**, flagging scores **above 0.6** for advisor follow-up.

**Dataset:** UCI "Predict Students' Dropout and Academic Success" (UCI id 697, 4,424 students, 36 features, target = Dropout / Enrolled / Graduate).

**Tech stack:** Kafka, Hadoop HDFS, classification model, advisor dashboard.

**Decisions made:** scikit-learn for the model, Streamlit for the dashboard, Docker Compose for the whole stack.

`/home/prajwal/Documents/BDA_LAB` is empty right now. The git repo it sits in is rooted at `/home/prajwal` (the home directory), so step 0 runs `git init` inside `BDA_LAB` and nothing gets committed to the home repo.

The first build step is to copy this plan into `BDA_LAB/PLAN.md`.

---

## Requirements

### Functional requirements
| ID | Requirement |
|----|-------------|
| FR1 | Download the UCI dataset (`data.csv`, `;`-separated) and give each student a `student_id`. |
| FR2 | **Weekly event simulation:** the dataset has semester totals, not weekly records. A producer turns each student's semester totals into week-by-week events over 2 semesters × 16 weeks = 32 weeks. A fixed random seed makes every run produce the same data. |
| FR3 | Three Kafka topics: `attendance`, `assignments`, `assessments`. Messages are JSON with `student_id` as the key, so all of one student's events go to the same partition. |
| FR4 | The producer sends one week at a time, with a configurable delay that speeds up time (e.g. 1 week = 10 s). Each topic gets a `week_end` marker after each week. |
| FR5 | A consumer reads all 3 topics and writes the events to HDFS at `/data/raw/<topic>/week=NN/part-*.jsonl`. When it sees `week_end` it writes a `_SUCCESS` file for that week. |
| FR6 | The static student profile (demographics, admission grade, scholarship, debtor, tuition up to date, etc.) is uploaded once to `/data/static/students.csv` on HDFS. |
| FR7 | **Feature engineering**, computed per student for each week *w* from all weeks up to *w*: attendance rate, assignment submission rate, average assessment score, number of failed assessments, a 3-week trend (recent weeks compared with earlier weeks), and the static profile features. |
| FR8 | **Training:** read the raw data from HDFS, build a feature snapshot for every (student, week) pair, and use the label Dropout = 1, Enrolled/Graduate = 0. Split by student, stratified, so no student appears in both train and test. Train a scikit-learn `HistGradientBoostingClassifier`. Save the model to HDFS at `/models/dropout_model.joblib`. |
| FR9 | **Evaluation:** report ROC-AUC, plus precision, recall and F1 at the **0.6** cutoff. Also report how early the model catches risk: the share of true dropouts flagged by weeks 4, 8 and 12. Save the metrics as JSON on HDFS. |
| FR10 | **Weekly scoring:** a scorer service checks HDFS for weeks where all 3 topics have `_SUCCESS`. For each week it hasn't scored yet, it writes `/data/scores/week=NN/scores.csv` with columns `student_id, week, risk_score, flagged` (flagged = `risk_score > 0.6`). |
| FR11 | **Advisor dashboard:** pick a week; see KPIs (students scored, number flagged, % flagged); see the flagged students sorted by risk; open one student to see their risk trend across weeks and their latest attendance, assignment and assessment numbers; filter by course; download the flagged list as CSV. |
| FR12 | The 0.6 threshold is set in one constant (`RISK_THRESHOLD`) that every part of the pipeline uses. |

### Non-functional requirements
- One command brings up the whole stack: `docker compose up`.
- The whole pipeline can be run again from scratch and gives the same results (fixed seed, scoring the same week twice overwrites the same file).
- It runs on this machine: 15 GB RAM, with the stack using about 6 GB or less.
- The dataset contains no real personal information: `student_id` is just the row number.
- The model choice and metrics are written up in the README for the lab report.

### Software / infrastructure requirements
- Docker 28 and Docker Compose v2 (already installed).
- **Kafka:** `apache/kafka:3.8.0`, a single broker in KRaft mode (no ZooKeeper).
- **HDFS:** `bde2020/hadoop-namenode:2.0.0-hadoop3.2.1-java8` and `bde2020/hadoop-datanode`. The namenode web UI is on port 9870, and Python talks to HDFS through WebHDFS.
- **Python 3.12 image (one image shared by all app services):** `kafka-python` 2.2, `hdfs` (the WebHDFS client), `pandas` 2.3, `scikit-learn` 1.7, `streamlit` 1.50 (pinned in `requirements.txt`). The dataset is downloaded with the standard library, so `ucimlrepo` is not needed.
- The app containers must run inside the Compose network. WebHDFS redirects to the datanode by hostname, and that hostname only resolves inside the network.

### Data requirements
- UCI dataset 697. It is downloaded once to `data/data.csv`, which is gitignored.
- The columns that drive the simulation: `Curricular units {1st,2nd} sem (enrolled / evaluations / approved / grade / without evaluations)`.
- There is **no real attendance data** in the dataset. The `Daytime/evening attendance` column is the class schedule, not presence. Attendance is simulated from the approved-to-enrolled ratio and the "without evaluations" count, with noise added. This limitation will be stated in the README.

---

## Architecture

```
data.csv ─► producer.py ─► Kafka [attendance | assignments | assessments]
                                   │
                                   ▼
                             consumer.py ─► HDFS /data/raw/<topic>/week=NN/
                                                      │
                 train.py (once) ◄──── features.py ◄──┤
                    │                                 │
                    ▼                                 ▼
          HDFS /models/dropout_model.joblib ─► score.py (polls weekly)
                                                      │
                                                      ▼
                                  HDFS /data/scores/week=NN/scores.csv
                                                      │
                                                      ▼
                                       dashboard.py (Streamlit :8501)
```

## Project layout

```
BDA_LAB/
  PLAN.md               # this plan
  README.md             # how to run + results + limitations
  docker-compose.yml    # kafka, namenode, datanode, producer, consumer, scorer, dashboard
  hadoop.env            # bde2020 HDFS config
  Dockerfile            # one python:3.12-slim image for all app services
  requirements.txt
  .gitignore            # data/, __pycache__
  src/
    common.py           # HDFS client, topic names, paths, RISK_THRESHOLD = 0.6
    download.py         # fetch UCI data, add student_id, upload static profile to HDFS
    producer.py         # simulate + publish weekly events
    consumer.py         # Kafka -> HDFS jsonl + _SUCCESS markers
    features.py         # raw events -> per-(student, week) feature table
    train.py            # train + evaluate + save model/metrics to HDFS
    score.py            # poll for completed weeks, write scores
    dashboard.py        # Streamlit advisor UI
  tests/
    test_features.py    # small assert-based checks on feature math + threshold flagging
```

## Build phases

0. **Setup.** Run `git init` inside `BDA_LAB`, then add `.gitignore`, `PLAN.md`, `requirements.txt` and `Dockerfile`.
1. **Infrastructure.** Write `docker-compose.yml` and `hadoop.env`. Check: the namenode UI is reachable at :9870, and a Kafka topic can be created and listed.
2. **Data.** `download.py` saves `data.csv` and puts `students.csv` on HDFS.
3. **Ingestion.** Write `producer.py` and `consumer.py`. Check: the `week=NN` folders appear in HDFS with `_SUCCESS` files.
4. **Features + model.** Write `features.py`, its tests, and `train.py`. Check: the model and metrics JSON are on HDFS, and ROC-AUC is sensible (this dataset usually gives around 0.85–0.9 for binary dropout).
5. **Weekly scoring.** Write `score.py` as a long-running Compose service. Check: a `scores.csv` file appears for each finished week.
6. **Dashboard.** Write `dashboard.py`. Check: it shows flagged students for the latest week, and the trend charts work.
7. **README.** Cover how to run it, the architecture diagram, the metrics, and the limitations (simulated weekly data and attendance).

## Verification (end-to-end)

1. Run `docker compose up -d kafka namenode datanode`, then `docker compose run --rm app python src/download.py`.
2. Run `docker compose up -d consumer producer` and watch `hdfs dfs -ls -R /data/raw` fill up week by week.
3. Run `docker compose run --rm app python src/train.py`. It prints ROC-AUC and P/R/F1 at 0.6, and `/models/dropout_model.joblib` exists on HDFS.
4. Run `docker compose up -d scorer dashboard`. `/data/scores/week=NN/` appears for each finished week.
5. Open `http://localhost:8501`. Every flagged row should have a risk above 0.6, and the flagged count should match the number of rows in `scores.csv` with `flagged == True`.
6. Run `python -m pytest tests/` or `python tests/test_features.py`. The tests pass.
