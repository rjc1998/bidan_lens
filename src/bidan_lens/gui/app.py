from __future__ import annotations

import time

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QApplication

from bidan_lens.config import AppConfig
from bidan_lens.diagnostics import LatencyRecorder
from bidan_lens.gui.popup import DictionaryPopup
from bidan_lens.gui.settings import SettingsDialog
from bidan_lens.gui.tray import TrayIcon
from bidan_lens.input import InputMonitor
from bidan_lens.models import PopupResult
from bidan_lens.paths import config_path
from bidan_lens.pipeline.coordinator import FrameRequest, PipelineCoordinator
from bidan_lens.screen import ScreenCapture


class ResultBridge(QObject):
    received = pyqtSignal(object)
    navigate = pyqtSignal(int)


class DesktopController:
    def __init__(
        self,
        application: QApplication,
        pipeline: PipelineCoordinator,
        latency_recorder: LatencyRecorder | None = None,
    ) -> None:
        self.application = application
        self.pipeline = pipeline
        self.latency_recorder = latency_recorder
        self.config = AppConfig.load(config_path())
        self.capture = ScreenCapture()
        self.popup = DictionaryPopup()
        self.bridge = ResultBridge()
        self.bridge.received.connect(self._show_result)
        self.bridge.navigate.connect(self.popup.navigate)
        self.pipeline.on_result = self.bridge.received.emit
        self.input = InputMonitor(
            self.config.activation_key,
            self.config.previous_result_key,
            self.config.next_result_key,
            lambda: self.bridge.navigate.emit(-1),
            lambda: self.bridge.navigate.emit(1),
        )
        self.enabled = True
        self._shutting_down = False
        self._last_scan_position: tuple[int, int] | None = None
        self._last_scan_at = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.setInterval(self.config.scan_interval_ms)
        self.tray = TrayIcon(self._toggle, self._settings, self.shutdown)

    def start(self) -> None:
        self.pipeline.start()
        self.input.start()
        self.timer.start()
        self.tray.show()

    def _tick(self) -> None:
        if not self.enabled or (not self.config.automatic_scanning and not self.input.active):
            self.popup.hide()
            return
        pointer = QCursor.pos()
        if self.popup.isVisible() and self.popup.geometry().contains(pointer):
            return
        position = (pointer.x(), pointer.y())
        now = time.monotonic()
        if self._last_scan_position is not None:
            distance = abs(position[0] - self._last_scan_position[0]) + abs(
                position[1] - self._last_scan_position[1]
            )
            if distance <= 2 and now - self._last_scan_at < 0.75:
                return
        self._last_scan_position = position
        self._last_scan_at = now
        if self.popup.isVisible() and not self.popup.capture_excluded:
            self.popup.hide()
            QApplication.processEvents()
        requested_at = time.monotonic()
        frame = self.capture.around(
            pointer.x(),
            pointer.y(),
            self.config.scan_width,
            self.config.scan_height,
        )
        self.pipeline.submit(
            FrameRequest(
                frame.image,
                frame.origin,
                (pointer.x(), pointer.y()),
                requested_at=requested_at,
            )
        )

    def _show_result(self, result: PopupResult | None) -> None:
        if result is None or not result.candidates:
            self.popup.hide()
            return
        pointer = QCursor.pos()
        if not result.target.box.contains(pointer.x(), pointer.y(), padding=2):
            self.popup.hide()
            return
        self.popup.show_result(result, pointer)
        if self.latency_recorder is not None and result.requested_at is not None:
            QApplication.processEvents()
            self.latency_recorder.record(result.requested_at)

    def _toggle(self, enabled: bool) -> None:
        self.enabled = enabled
        if not enabled:
            self.popup.hide()

    def _settings(self) -> None:
        dialog = SettingsDialog(self.config)
        if dialog.exec():
            dialog.apply()
            self.config.save(config_path())
            self.timer.setInterval(self.config.scan_interval_ms)
            self.input.activation_key = self.input._canonical_name(self.config.activation_key)
            self.input.previous_result_keys = self.input._parse_chord(
                self.config.previous_result_key
            )
            self.input.next_result_keys = self.input._parse_chord(self.config.next_result_key)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        self.timer.stop()
        self.input.stop()
        self.pipeline.stop()
        self.capture.close()
        self.tray.hide()
        self.application.quit()
