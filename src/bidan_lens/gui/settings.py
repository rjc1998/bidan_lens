from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
)

from bidan_lens.config import AppConfig


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.setWindowTitle("BiDan Lens settings")
        self.config = config
        form = QFormLayout(self)
        self.automatic = QCheckBox("Scan automatically while the pointer rests on text")
        self.automatic.setChecked(config.automatic_scanning)
        self.activation = QLineEdit(config.activation_key)
        self.previous = QLineEdit(config.previous_result_key)
        self.next = QLineEdit(config.next_result_key)
        self.interval = QSpinBox()
        self.interval.setRange(100, 2000)
        self.interval.setValue(config.scan_interval_ms)
        self.results = QSpinBox()
        self.results.setRange(1, 10)
        self.results.setValue(config.max_results)
        form.addRow(self.automatic)
        form.addRow("Hold key (manual mode)", self.activation)
        form.addRow("Previous result hotkey", self.previous)
        form.addRow("Next result hotkey", self.next)
        form.addRow("Scan interval (ms)", self.interval)
        form.addRow("Maximum analyses", self.results)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def apply(self) -> None:
        self.config.automatic_scanning = self.automatic.isChecked()
        self.config.activation_key = self.activation.text().strip().lower() or "shift"
        self.config.previous_result_key = self.previous.text().strip().lower() or "ctrl+shift+up"
        self.config.next_result_key = self.next.text().strip().lower() or "ctrl+shift+down"
        self.config.scan_interval_ms = self.interval.value()
        self.config.max_results = self.results.value()
