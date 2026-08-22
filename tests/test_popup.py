import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from bidan_lens.gui.popup import DictionaryPopup
from bidan_lens.models import (
    AnalysisCandidate,
    BoundingBox,
    HoverTarget,
    PopupResult,
)


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def test_popup_navigates_candidates_and_copies_learner_lemma(application) -> None:
    target = HoverTarget("먹어요", "먹어요", 0, 3, BoundingBox(0, 0, 50, 20), 0.9)
    result = PopupResult(
        target,
        (
            AnalysisCandidate("먹어요", "먹다", 1.0),
            AnalysisCandidate("먹어요", "먹이다", 0.8, uncertain=True),
        ),
    )
    popup = DictionaryPopup()
    popup._result = result
    popup._render()
    assert popup.lemma.text() == "Dictionary form: 먹다"
    popup.navigate(1)
    assert popup.lemma.text() == "Dictionary form: 먹이다"
    popup._copy_lemma()
    assert application.clipboard().text() == "먹이다"
