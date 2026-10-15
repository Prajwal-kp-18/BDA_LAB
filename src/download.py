"""Fetch the UCI dropout dataset, add student_id, and upload the static profile to HDFS."""
import io
import pathlib
import urllib.request
import zipfile

import pandas as pd

from common import STATIC, exists, hdfs, write_csv

URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
LOCAL = pathlib.Path(__file__).resolve().parent.parent / "data" / "data.csv"


def load_local():
    """The institution's source records (curricular data drives the event simulation)."""
    df = pd.read_csv(LOCAL, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # raw header has a stray tab in one column name
    df.insert(0, "student_id", range(1, len(df) + 1))
    return df


def static_profile(df):
    # semester outcomes are excluded: they would leak the very results we try to predict early
    return df.drop(columns=[c for c in df.columns if c.startswith("Curricular units")])


if __name__ == "__main__":
    if not LOCAL.exists():
        LOCAL.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extract("data.csv", LOCAL.parent)
    df = load_local()
    client = hdfs()
    if exists(client, STATIC):  # don't rewrite under a trainer that may be reading it
        print(f"{STATIC} already in HDFS")
    else:
        write_csv(client, STATIC, static_profile(df))
        print(f"{len(df)} students; static profile uploaded to {STATIC}")
