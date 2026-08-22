from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from bidan_lens.models import PopupResult
from bidan_lens.windows import exclude_window_from_capture


class DictionaryPopup(QFrame):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("dictionaryPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setWindowFlag(Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setMaximumWidth(500)
        self._result: PopupResult | None = None
        self._capture_excluded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        self.surface = QLabel()
        self.surface.setObjectName("surface")
        self.lemma = QLabel()
        self.definitions = QLabel()
        self.definitions.setWordWrap(True)
        self.breakdown = QLabel()
        self.breakdown.setWordWrap(True)
        self.correction = QLabel()
        self.correction.setWordWrap(True)
        layout.addWidget(self.surface)
        layout.addWidget(self.correction)
        layout.addWidget(self.lemma)
        layout.addWidget(self.definitions)
        layout.addWidget(self.breakdown)
        controls = QHBoxLayout()
        self.position = QLabel()
        copy_surface = QPushButton("Copy word")
        copy_lemma = QPushButton("Copy dictionary form")
        copy_surface.clicked.connect(self._copy_surface)
        copy_lemma.clicked.connect(self._copy_lemma)
        controls.addWidget(self.position)
        controls.addStretch()
        controls.addWidget(copy_surface)
        controls.addWidget(copy_lemma)
        layout.addLayout(controls)
        self.setStyleSheet(
            """
            QFrame#dictionaryPopup { background: #171a20; color: #f3f5f7;
              border: 1px solid #596271; border-radius: 9px; }
            QLabel#surface { font-size: 22px; font-weight: 700; }
            QLabel { font-size: 13px; }
            QPushButton { background: #2c3340; border: 0; border-radius: 4px;
              padding: 4px 7px; color: #e8eaed; }
            """
        )

    @property
    def capture_excluded(self) -> bool:
        return self._capture_excluded

    def show_result(self, result: PopupResult, pointer: QPoint) -> None:
        self._result = result
        self._render()
        self.adjustSize()
        screen = QGuiApplication.screenAt(pointer) or QGuiApplication.primaryScreen()
        available = screen.availableGeometry()
        x = min(pointer.x() + 18, available.right() - self.width())
        y = pointer.y() + 24
        if y + self.height() > available.bottom():
            y = pointer.y() - self.height() - 18
        self.move(max(available.left(), x), max(available.top(), y))
        self.show()
        if not self._capture_excluded:
            self._capture_excluded = exclude_window_from_capture(int(self.winId()))

    def navigate(self, delta: int) -> None:
        if not self._result or not self._result.candidates:
            return
        self._result = self._result.with_index(self._result.selected_index + delta)
        self._render()
        self.adjustSize()

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.navigate(-1 if event.angleDelta().y() > 0 else 1)
        event.accept()

    def _render(self) -> None:
        assert self._result is not None
        candidate = self._result.selected
        self.surface.setText(self._result.target.surface)
        if candidate is None:
            self.lemma.setText("No dictionary analysis found")
            self.definitions.clear()
            self.breakdown.clear()
            self.correction.clear()
            self.position.clear()
            return
        self.correction.setText(
            f"Interpreted as: {candidate.interpreted_surface}"
            if candidate.interpreted_surface and candidate.interpreted_surface != candidate.surface
            else ""
        )
        self.lemma.setText(f"Dictionary form: {candidate.lemma}")
        definitions = [
            sense.definition for entry in candidate.dictionary_entries for sense in entry.senses
        ]
        self.definitions.setText(
            "\n".join(
                f"{index}. {definition}" for index, definition in enumerate(definitions[:5], 1)
            )
            or "No English definition found"
        )
        parts = [f"{part.surface}: {part.learner_label}" for part in candidate.morphemes]
        parts.extend(f"{feature.label}: {feature.explanation}" for feature in candidate.features)
        self.breakdown.setText("\n".join(parts))
        uncertainty = " · uncertain" if candidate.uncertain else ""
        self.position.setText(
            f"{self._result.selected_index + 1}/{len(self._result.candidates)}{uncertainty}"
        )

    def _copy_surface(self) -> None:
        if self._result:
            QGuiApplication.clipboard().setText(self._result.target.surface)

    def _copy_lemma(self) -> None:
        if self._result and self._result.selected:
            QGuiApplication.clipboard().setText(self._result.selected.lemma)
