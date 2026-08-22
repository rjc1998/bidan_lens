from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from bidan_lens.assets import AssetError, AssetManager


class SetupDialog(QDialog):
    def __init__(self, manager: AssetManager, parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.manager = manager
        self.installed_path: Path | None = None
        self.setWindowTitle("Set up BiDan Lens")
        self.setMinimumWidth(500)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "BiDan Lens needs Korean OCR models and a local English KRDict database. "
            "The bundle is downloaded only when you choose Download. Screenshots are "
            "never uploaded. You can instead import a verified bundle from disk."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.url = QLineEdit(os.environ.get("BIDAN_LENS_ASSET_URL", ""))
        self.url.setPlaceholderText("Release asset bundle URL")
        layout.addWidget(self.url)
        download = QPushButton("Download and install")
        import_bundle = QPushButton("Import offline bundle…")
        cancel = QPushButton("Exit setup")
        download.clicked.connect(self._download)
        import_bundle.clicked.connect(self._import)
        cancel.clicked.connect(self.reject)
        layout.addWidget(download)
        layout.addWidget(import_bundle)
        layout.addWidget(cancel)

    def _download(self) -> None:
        url = self.url.text().strip()
        if not url:
            QMessageBox.warning(self, "Missing URL", "Enter the published asset bundle URL.")
            return
        self._install(lambda: self.manager.download_and_install(url))

    def _import(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self, "Import BiDan Lens assets", "", "BiDan Lens bundles (*.zip)"
        )
        if selected:
            self._install(lambda: self.manager.install_bundle(Path(selected)))

    def _install(self, operation) -> None:  # type: ignore[no-untyped-def]
        try:
            self.installed_path = operation()
        except (AssetError, OSError) as error:
            QMessageBox.critical(self, "Asset installation failed", str(error))
            return
        QMessageBox.information(self, "Setup complete", "Assets were verified and installed.")
        self.accept()
