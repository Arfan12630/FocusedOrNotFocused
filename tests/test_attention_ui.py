import json
import time

import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from brainviz.attention import AttentionDialog
from brainviz.attention_protocol import QualityConfig
from brainviz.sim_source import SimSource
from brainviz.source import Channel, State, Status
from brainviz.ui import MainWindow


@pytest.fixture
def attention(tmp_path):
    app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win=MainWindow(SimSource(seed=3),skip_impedance=True,out_dir=tmp_path/'legacy')
    win.hub.configure([Channel(0,'Ch1'),Channel(1,'Ch2')],250)
    win._setup_channels();win.status=Status(State.STREAMING)
    dialog=AttentionDialog(win,root=tmp_path/'data',config=QualityConfig(preview_s=1),practice_trials=2,measured_trials=10)
    win._attention_dialog=dialog
    yield app,win,dialog
    dialog.finish_session('aborted');win.close()


def prepare(win,dialog):
    for name in ('arrangement','reference','ground'): getattr(dialog,name).setText('explicit test setup')
    win._last_impedance=np.array([10000.,20000.])
    win._last_impedance_time=time.perf_counter()
    win.hub.push(np.arange(500),np.zeros(500),np.random.default_rng(1).normal(0,4,(500,2)))
    dialog.preview_start=time.perf_counter()-2
    dialog.tick()


def test_preflight_requires_setup_impedance_and_fresh_samples(attention):
    _,win,d=attention
    d.tick();assert not d.start_button.isEnabled()
    prepare(win,d);d._last_preview_update=0;d.tick()
    assert d.start_button.isEnabled()
    win.hub.last_receipt=time.perf_counter()-10;d._last_preview_update=0;d.tick()
    assert not d.start_button.isEnabled()


def test_partial_run_closes_raw_and_events_and_cannot_be_labelled(attention):
    _,win,d=attention
    prepare(win,d);d._last_preview_update=0;d.tick();d.start_session()
    assert d.active and win.recorder is d.rec
    win.hub.push(np.arange(500,1000),np.zeros(500),np.random.default_rng(2).normal(0,4,(500,2)))
    d.finish_session('aborted')
    assert win.recorder is None and not d.active
    meta=json.loads((d.directory/'session.json').read_text())
    assert meta['state']=='aborted' and d.summary['training_eligible_windows']==0
    assert (d.directory/d.rec.packet_path.name).is_file()
    assert (d.directory/'events.jsonl').read_text().count('session_end')==1
    d.finish_session('aborted')  # idempotent
    assert (d.directory/'events.jsonl').read_text().count('session_end')==1


def test_disconnect_and_stale_stream_preserve_partial_data(attention):
    _,win,d=attention
    prepare(win,d);d._last_preview_update=0;d.tick();d.start_session()
    assert d.active
    win.hub.last_receipt=time.perf_counter()-2
    d.tick()
    assert d.meta['state']=='interrupted' and win.recorder is None


def test_recording_start_failure_does_not_leave_active_task(attention,monkeypatch):
    _,win,d=attention
    prepare(win,d);d._last_preview_update=0;d.tick()
    def fail(*a,**k): raise OSError('disk full')
    monkeypatch.setattr(win,'_start_recording',fail)
    d.start_session()
    assert not d.active and win.recorder is None
    assert 'disk full' in d.message.text()


def test_focus_loss_interrupts_run(attention):
    _,win,d=attention
    prepare(win,d);d._last_preview_update=0;d.tick();d.start_session()
    d.started=time.perf_counter()-1
    d.changeEvent(QtCore.QEvent(QtCore.QEvent.Type.ActivationChange))
    assert not d.active and d.meta['state']=='interrupted'


def test_ch2_only_ignores_bad_ch1_and_writes_actual_ch2_values(attention):
    from brainviz.recorder import read_csv
    _,win,d=attention
    d.sync_channels();d.channel.setCurrentIndex(d.channel.findData(1))
    prepare(win,d)
    rng=np.random.default_rng(42)
    good=rng.normal(0,3,500)
    win.hub.push(np.arange(500,1000),np.zeros(500),np.column_stack([np.full(500,800000.),good]))
    d._last_preview_update=0;d.tick()
    assert d.start_button.isEnabled() and d.quality['valid']
    d.start_session()
    assert d.active and d.rec.labels==['Ch2'] and d.meta['recorded_channels']==['Ch2']
    win.hub.push(np.arange(1000,1500),np.zeros(500),np.column_stack([np.full(500,800000.),good]))
    d.finish_session('aborted')
    header,data=read_csv(d.rec.path)
    assert header==['sample_index','host_time_unix','pack_num','marker','Ch2']
    assert data.shape==(500,5)
    np.testing.assert_allclose(data[:,4],good,atol=.00051)
    meta=json.loads(d.rec.sidecar.read_text())
    assert meta['channels'][0]['num']==1 and meta['source_channel_indices']==[1]


def test_start_without_checks_allows_blank_setup_and_bad_quality(attention):
    _,win,d=attention
    win.hub.push(np.arange(500),np.zeros(500),np.full((500,2),800000.))
    d.preview_start=time.perf_counter()-2
    d._last_preview_update=0;d.tick()
    assert not d.start_button.isEnabled() and d.bypass_button.isEnabled()
    d.start_session(bypass=True)
    assert d.active and d.meta['checks_skipped']
    assert all(d.meta['setup'][name]=='unknown' for name in ('reference','ground','arrangement'))
    assert d.meta['impedance_ohms'] is None
    d.finish_session('aborted')
    assert d.summary['checks_skipped'] and d.summary['state']=='aborted'


def test_bypass_still_requires_fresh_live_samples(attention):
    _,win,d=attention
    d.start_session(bypass=True)
    assert not d.active and d.directory is None
    win.hub.push(np.arange(10),np.zeros(10),np.ones((10,2)))
    win.hub.last_receipt=time.perf_counter()-10
    d.start_session(bypass=True)
    assert not d.active and d.directory is None
