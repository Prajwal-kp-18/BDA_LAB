"""Run: python tests/test_features.py  (or pytest)"""
import math
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from common import flag  # noqa: E402
from features import build, snapshot  # noqa: E402

RAW = {
    "attendance": pd.DataFrame({"student_id": [1, 1, 1, 1], "week": [1, 2, 3, 4],
                                "sessions": [5, 5, 5, 5], "attended": [5, 5, 1, 1]}),
    "assignments": pd.DataFrame({"student_id": [1, 1, 1, 1], "week": [1, 2, 3, 4],
                                 "assigned": [2, 2, 2, 2], "submitted": [2, 2, 0, 0]}),
    "assessments": pd.DataFrame({"student_id": [1, 1, 1], "week": [2, 4, 5],
                                 "score": [14.0, 6.0, 18.0], "passed": [1, 0, 1]}),
}


def test_snapshot_uses_only_past_weeks():
    f = snapshot(RAW, 4, [1, 2]).set_index("student_id")
    s = f.loc[1]
    assert s.att_rate == 12 / 20
    assert math.isclose(s.att_trend, 7 / 15 - 12 / 20)  # weeks 2-4 vs all
    assert s.sub_rate == 4 / 8
    assert s.n_assessments == 2 and s.n_failed == 1 and s.avg_score == 10.0  # week-5 exam not leaked
    assert s.fail_rate == 0.5


def test_student_without_events():
    s = snapshot(RAW, 1, [1, 2]).set_index("student_id").loc[2]
    assert s.n_assessments == 0 and s.n_failed == 0
    assert math.isnan(s.att_rate) and math.isnan(s.avg_score)


def test_build_drops_label():
    static = pd.DataFrame({"student_id": [1, 2], "Age at enrollment": [19, 30], "Target": ["Dropout", "Graduate"]})
    df = build(RAW, [1, 2], static)
    assert len(df) == 4 and "Target" not in df and "Age at enrollment" in df


def test_threshold_is_strict():
    assert not flag(0.6) and flag(0.6001) and not flag(0.2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all feature tests passed")
