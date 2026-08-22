from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from bidan_lens.models import PopupResult
from bidan_lens.windows import exclude_window_from_capture

_POS_LABELS = {
    'verb': 'action verb',
    'adjective': 'descriptive verb',
    'noun': 'noun',
    '보조 동사': 'helping verb',
    '보조 형용사': 'helping descriptive verb',
}
_LEXICAL_ROLES = {
    'action verb',
    'descriptive verb',
    'helping verb',
    'noun',
    'name or proper noun',
    'dependent noun',
    'pronoun',
    'number',
}


def _definitions_text(candidate) -> str:  # type: ignore[no-untyped-def]
    component = candidate.lexical_components[0] if candidate.lexical_components else None
    role = component.learner_role if component else 'dictionary entry'
    explanation = component.contextual_explanation if component else None
    lines = [f'Role in this sentence: {role}']
    if explanation:
        lines[0] += f' — {explanation}'
    entries = candidate.dictionary_entries
    if not entries:
        lines.append('No English definition found')
        return '\n'.join(lines)
    leading_pos = entries[0].part_of_speech
    shown = 0
    for entry in entries:
        group = 'Matching dictionary entry' if entry.part_of_speech == leading_pos else 'Other use'
        label = _POS_LABELS.get(entry.part_of_speech or '', entry.part_of_speech or 'entry')
        homograph = f' {entry.homograph_number}' if entry.homograph_number else ''
        lines.append(f'{group} — {label}{homograph}')
        for sense in entry.senses:
            if shown == 5:
                return '\n'.join(lines)
            shown += 1
            lines.append(f'{shown}. {sense.definition}')
    return '\n'.join(lines)


def _breakdown_text(candidate) -> str:  # type: ignore[no-untyped-def]
    lines: list[str] = []
    if candidate.lexical_components:
        lines.append('Components')
    for component in candidate.lexical_components:
        gloss = next(
            (
                sense.definition
                for entry in component.dictionary_entries
                for sense in entry.senses
            ),
            None,
        )
        value = f'{component.surface} → {component.lemma}: {component.learner_role}'
        lines.append(f'{value} — {gloss}' if gloss else value)
    grammar = [
        part
        for part in candidate.morphemes
        if part.learner_label not in _LEXICAL_ROLES
    ]
    if grammar or candidate.features:
        lines.append('Grammar')
    lines.extend(f'{part.surface}: {part.learner_label}' for part in grammar)
    lines.extend(
        f'{feature.label}: {feature.explanation}' for feature in candidate.features
    )
    return '\n'.join(lines)


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
        if candidate.interpreted_surface and candidate.interpreted_surface != candidate.surface:
            self.correction.setText(f'Spacing: {candidate.interpreted_surface}')
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
        self.definitions.setText(_definitions_text(candidate))
        self.breakdown.setText(_breakdown_text(candidate))
        uncertainty = " · uncertain" if candidate.uncertain else ""
        uncertainty = ' · uncertain' if candidate.uncertain else ''
        self.position.setText(
            f"{self._result.selected_index + 1}/{len(self._result.candidates)}{uncertainty}"
        )

    def _copy_surface(self) -> None:
        if self._result:
            QGuiApplication.clipboard().setText(self._result.target.surface)

    def _copy_lemma(self) -> None:
        if self._result and self._result.selected:
            QGuiApplication.clipboard().setText(self._result.selected.lemma)
