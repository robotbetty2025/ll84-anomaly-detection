"""Smoke tests: every Streamlit page renders without an exception, and the scoring form works."""
import sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
sys.path.insert(0, str(APP))


def run(page):
    at = AppTest.from_file(str(APP / page), default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_pages_render():
    for p in ["Home.py", "pages/1_Resume.py", "pages/2_Projects.py", "pages/3_Anomaly_Detector.py"]:
        run(p)


def test_score_form():
    at = run("pages/3_Anomaly_Detector.py")
    at.button[0].click().run(timeout=180)       # form submit button
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Actual site EUI" in labels and "Expected EUI (LightGBM)" in labels, labels


def test_filters_empty_selection():
    at = run("pages/3_Anomaly_Detector.py")
    at.sidebar.multiselect[0].set_value([]).run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]


if __name__ == "__main__":
    test_pages_render(); test_score_form(); test_filters_empty_selection(); print("all app tests passed")
