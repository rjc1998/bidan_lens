from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon


def _icon() -> QIcon:
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("transparent"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor("#5bb7a8"))
    painter.setPen(QColor("#eaf9f6"))
    painter.drawEllipse(3, 3, 58, 58)
    painter.drawText(pixmap.rect(), 0x84, "한")
    painter.end()
    return QIcon(pixmap)


class TrayIcon(QSystemTrayIcon):
    def __init__(
        self,
        on_toggle: Callable[[bool], None],
        on_settings: Callable[[], None],
        on_quit: Callable[[], None],
    ) -> None:
        super().__init__(_icon())
        self.setToolTip("BiDan Lens")
        menu = QMenu()
        self.enabled = QAction("Scanning enabled", menu)
        self.enabled.setCheckable(True)
        self.enabled.setChecked(True)
        self.enabled.toggled.connect(on_toggle)
        settings = menu.addAction("Settings…")
        settings.triggered.connect(on_settings)
        menu.addSeparator()
        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(on_quit)
        menu.insertAction(settings, self.enabled)
        self.setContextMenu(menu)
