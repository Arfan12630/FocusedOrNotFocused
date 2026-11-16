"""Real-headset quality pilot only; never generates attention labels or fake responses.

Run from the project root: .venv/bin/python -P scripts/headset_preflight.py
Close the visualizer/NeuroREC first; one BLE client at a time.
"""
import json
from pathlib import Path
import sys
import threading
import time

sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
from brainviz.sdk_source import SDKSource
from brainviz.source import State
from brainviz.ringbuffer import DataHub
from brainviz.recorder import Recorder
from brainviz.quality import assess_quality
from brainviz.attention_protocol import QualityConfig
from brainviz.attention_store import DATA_ROOT, new_session_dir
from brainviz.attention_analysis import write_json

source=SDKSource(scan_timeout=15)
hub=DataHub()
resistance=[]
chunks=[]
source.on_ready=hub.configure
source.on_status=lambda st: print(st.state.value, st.message, flush=True)
source.on_impedance=lambda x: resistance.append(np.array(x,copy=True))
def data(pack,marker,uv):
    hub.push(pack,marker,uv)
    if hub.recorder is not None:
        chunks.append((np.array(pack,copy=True),np.array(uv,copy=True)))
source.on_data=data

def until(condition,timeout,description):
    end=time.perf_counter()+timeout
    while time.perf_counter()<end:
        if condition(): return
        if source.status.state is State.ERROR:
            raise RuntimeError(source.status.message)
        time.sleep(.05)
    raise RuntimeError('Timed out: '+description)

rec=None
try:
    source.start()
    until(lambda:source.status.state is State.STREAMING,45,'connect')
    source.start_impedance()
    until(lambda:bool(resistance),10,'impedance readings')
    time.sleep(2)
    source.stop_impedance()
    until(lambda:source.status.state is State.STREAMING,10,'signal resume')
    directory=new_session_dir(DATA_ROOT/'headset_pilots')
    config=QualityConfig()
    rec=Recorder(directory,[c.label for c in hub.channels],hub.fs,
        {'task':{'name':'headset_quality_pilot','attention_labels':'none'},'device':source.info},packet_timing=True)
    rec.start();hub.recorder=rec
    start=time.perf_counter()
    while time.perf_counter()-start<10:
        if source.status.state is not State.STREAMING or rec.error:
            raise RuntimeError(rec.error or 'Headset disconnected')
        time.sleep(.05)
    hub.recorder=None;rec.stop()
    pack=np.concatenate([c[0] for c in chunks]) if chunks else np.empty(0)
    raw=np.concatenate([c[1] for c in chunks]) if chunks else np.empty((0,len(hub.channels)))
    quality=assess_quality(raw,hub.fs,pack,source.info.get('gain',3),config)
    report={'kind':'headset_quality_pilot','attention_labels':'none','quality_config':config.to_dict(),
            'quality':quality,'samples':len(raw),'duration_s':time.perf_counter()-start,
            'impedance_ohms':[float(v) if np.isfinite(v) else None for v in resistance[-1]],
            'measured_rate_hz':hub.rate.hz,'limitation':'Checks provisional artifacts only; not confirmation of brain activity.'}
    write_json(directory/'pilot_report.json',report)
    print(json.dumps(report,indent=2),flush=True)
    print('Saved',directory,flush=True)
    raise SystemExit(0 if quality['valid'] else 2)
finally:
    hub.recorder=None
    if rec is not None and rec._thread is not None:
        rec.stop()
    source.stop()
