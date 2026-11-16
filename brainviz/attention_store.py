"""Local-only session storage and explicitly requested ZIP exports."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import uuid
import zipfile

from .attention_analysis import write_json

DATA_ROOT = Path(os.environ.get("BRAINVIZ_DATA_DIR", str(Path.cwd() / "participant_data")))
SENSITIVE_KEYS = {"serial", "serial_number", "serialnumber", "address", "bluetooth_address", "ble_address"}


def new_session_dir(root=DATA_ROOT, simulated=False):
    name = time.strftime("%Y%m%d-%H%M%S") + ("-SIM" if simulated else "") + "-" + uuid.uuid4().hex[:8]
    directory = Path(root) / name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def sanitized(value):
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items() if key.lower() not in SENSITIVE_KEYS}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    return value


def export_session(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination)
    meta = json.loads((directory / "session.json").read_text())
    if meta["state"] == "running":
        raise ValueError("Stop the session before exporting")
    eeg = Path(meta["eeg_file"])
    packets = Path(meta["packet_timing_file"])
    allowed = ["session.json", "events.jsonl", "sequence.json", "summary.json", "trials.csv", "windows.csv",
               str(eeg), str(eeg.with_suffix(".json")), str(packets)]
    if any(Path(name).name != name for name in allowed):
        raise ValueError("Session manifest contains an unsafe path")
    if destination.exists():
        raise FileExistsError("Choose a new ZIP filename; existing exports are not overwritten")
    if destination.resolve().parent == directory:
        raise ValueError("Save exports outside the raw session folder")
    temporary = destination.with_name(destination.name + ".tmp")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in dict.fromkeys(allowed):
                path = directory / name
                if path.is_symlink():
                    raise ValueError("Symlinks are not exported")
                if not path.is_file():
                    # Interrupted runs can be exported even if analysis failed.
                    if name in ("trials.csv", "windows.csv", "summary.json"):
                        continue
                    raise FileNotFoundError(path.name)
                if path.suffix == ".json":
                    content = json.dumps(sanitized(json.loads(path.read_text())), indent=2) + "\n"
                    archive.writestr(name, content)
                else:
                    archive.write(path, name)
            archive.writestr("EXPORT-NOTE.txt", "Manual local export. Device serial/Bluetooth identifier fields "
                             "are omitted from JSON metadata. EEG, responses and timestamps remain personal data.\n")
        temporary.replace(destination)
    except Exception:
        if created:
            temporary.unlink(missing_ok=True)
        raise
    return destination
