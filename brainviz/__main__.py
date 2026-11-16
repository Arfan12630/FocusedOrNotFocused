"""CLI: python -m brainviz [--simulate] ..."""
from __future__ import annotations

import argparse
import signal
import sys

GAIN_CHOICES = ["none", "1", "2", "3", "4", "5", "6", "8", "12", "24"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="brainviz", description="Real-time BrainBit Flex8 EEG visualizer")
    p.add_argument("--simulate", action="store_true",
                   help="generate fake EEG instead of connecting to the headset")
    p.add_argument("--arithmetic", action="store_true", help="open an arithmetic task with EEG recording")
    p.add_argument("--gradcpt", action="store_true", help="open the local attention-task collection workflow")
    p.add_argument("--record-channel", type=int, metavar="N", help="preselect one attention recording channel (1-based)")
    p.add_argument("--quality-config", metavar="JSON", help="versioned provisional attention-quality settings")
    p.add_argument("--gain", choices=GAIN_CHOICES, default="3",
                   help="amplifier gain written to every channel (default 3; 'none' leaves the device as-is)")
    p.add_argument("--notch", type=float, default=60.0, metavar="HZ",
                   help="mains notch frequency (default 60; 0 disables)")
    p.add_argument("--labels", default="", metavar="A,B,...",
                   help="comma-separated channel labels, applied in channel order")
    p.add_argument("--skip-impedance", action="store_true", help="don't open the impedance check after connecting")
    p.add_argument("--out", default="recordings", metavar="DIR", help="recording directory (default ./recordings)")
    p.add_argument("--seed", type=int, default=None, help="random seed for --simulate")
    args = p.parse_args(argv)
    if args.arithmetic and args.gradcpt:
        p.error("choose either --arithmetic or --gradcpt")
    if args.record_channel is not None and args.record_channel < 1:
        p.error("--record-channel must be at least 1")
    if args.quality_config:
        import json
        from pathlib import Path
        from .attention_protocol import QualityConfig
        try:
            args.attention_quality_config = QualityConfig(**json.loads(Path(args.quality_config).read_text()))
        except (OSError, ValueError, TypeError) as exc:
            p.error(f"invalid quality config: {exc}")
    return args


def build_source(args):
    if args.simulate:
        from .sim_source import SimSource
        source = SimSource(seed=args.seed)
    else:
        from .sdk_source import SDKSource
        source = SDKSource(gain=None if args.gain == "none" else int(args.gain))
    source.label_overrides = [s.strip() for s in args.labels.split(",")] if args.labels else []
    return source


def main(argv=None) -> int:
    args = parse_args(argv)
    from PyQt6 import QtCore, QtWidgets
    import pyqtgraph as pg
    from .ui import MainWindow, SURFACE, MUTED, apply_theme

    pg.setConfigOptions(antialias=False, background=SURFACE, foreground=MUTED)
    app = QtWidgets.QApplication(sys.argv[:1])
    apply_theme(app)
    window = MainWindow(build_source(args), out_dir=args.out, notch=args.notch or None,
                        skip_impedance=args.skip_impedance)
    app.aboutToQuit.connect(window.close)
    window.attention_quality_config = getattr(args, "attention_quality_config", None)
    window.attention_record_channel = args.record_channel
    window.show()
    window.start()
    if args.arithmetic:
        window.open_arithmetic()
    if args.gradcpt:
        window.open_attention()

    # Let Ctrl+C in the terminal close the app cleanly: Python only sees signals while the interpreter runs.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    keepalive = QtCore.QTimer()
    keepalive.start(250)
    keepalive.timeout.connect(lambda: None)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
