from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bidan_lens import __version__
from bidan_lens.analysis.korean import KoreanAnalyzer
from bidan_lens.assets import AssetManager
from bidan_lens.diagnostics import LatencyRecorder
from bidan_lens.dictionary.store import SqliteDictionaryStore
from bidan_lens.ocr.paddle import PaddleOcrEngine
from bidan_lens.paths import assets_path
from bidan_lens.pipeline.coordinator import PipelineCoordinator
from bidan_lens.windows import enable_per_monitor_dpi_awareness


def _arguments(argv: list[str]) -> tuple[Path | None, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--latency-report", type=Path)
    arguments, remaining = parser.parse_known_args(argv[1:])
    return arguments.latency_report, [argv[0], *remaining]


def main() -> int:
    if "--version" in sys.argv:
        print(f"BiDan Lens {__version__}")
        return 0
    latency_report, qt_arguments = _arguments(sys.argv)
    enable_per_monitor_dpi_awareness()
    from PyQt6.QtWidgets import QApplication, QDialog

    from bidan_lens.gui.app import DesktopController
    from bidan_lens.gui.setup_dialog import SetupDialog

    application = QApplication(qt_arguments)
    application.setApplicationName("BiDan Lens")
    application.setQuitOnLastWindowClosed(False)
    manager = AssetManager(assets_path())
    asset_directory = manager.current_dir()
    if asset_directory is None:
        setup = SetupDialog(manager)
        if setup.exec() != QDialog.DialogCode.Accepted or setup.installed_path is None:
            return 1
        asset_directory = setup.installed_path

    dictionary = SqliteDictionaryStore(asset_directory / "dictionary.sqlite3")
    analyzer = KoreanAnalyzer(dictionary)
    ocr = PaddleOcrEngine.from_asset_directory(asset_directory)
    pipeline = PipelineCoordinator(ocr, analyzer, lambda _result: None)
    latency_recorder = (
        LatencyRecorder(bundle_version=asset_directory.name) if latency_report else None
    )
    controller = DesktopController(application, pipeline, latency_recorder)
    application.aboutToQuit.connect(controller.shutdown)
    controller.start()
    exit_code = application.exec()
    if latency_report is not None and latency_recorder is not None:
        latency_recorder.write(latency_report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
