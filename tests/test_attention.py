import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from brainviz.attention_protocol import make_sequence, QualityConfig
from brainviz.attention_analysis import compute_vtc, window_label, sample_times, process_session, write_json
from brainviz.attention_store import export_session
from brainviz.gradcpt_scoring import assign_trial, score_trials
from brainviz.quality import assess_quality
from brainviz.recorder import Recorder


def test_sequence_reproducible_and_protocol_lengths():
    a = make_sequence(16)
    assert a == make_sequence(16) and a != make_sequence(17)
    for phase, count in [('practice', 150), ('measured', 600)]:
        rows = [t for t in a if t['phase'] == phase]
        assert len(rows) == count and sum(t['category'] == 'mountain' for t in rows) == count // 10
        assert all(x['image'] != y['image'] for x, y in zip(rows, rows[1:]))
        assert sum(t['duration_s'] for t in rows) == pytest.approx(count * .8)


@pytest.mark.parametrize('op,prev,cur,pg,cg,expected', [
    (.2,False,False,1,1,(-1,1)), (.9,False,False,1,1,(0,2)),
    (.5,True,False,1,1,(0,3)), (.5,False,True,1,1,(-1,4)),
    (.5,False,False,1,0,(-1,5)), (.5,False,False,0,1,(0,6)),
    (.7,False,False,1,1,(0,7)), (.5,False,False,1,1,(-1,8)),
    (.5,False,False,2,0,(0,10)), (.5,True,True,1,1,None),
    (.2,True,False,1,1,None), (.9,False,True,1,1,None),
])
def test_reference_assignment_branches(op,prev,cur,pg,cg,expected):
    assert assign_trial(op,prev,cur,pg,cg) == expected


def test_delayed_response_and_code4_chronological_swap():
    rows = [{'trial_id':i,'category':'city','onset_s':i*.8,'duration_s':.8} for i in range(3)]
    # No response to previous; early ambiguous press assigned current, second swaps them.
    result = score_trials(rows,[{'time_s':1.36,'trial_id':1,'opacity':.7},
                               {'time_s':1.40,'trial_id':1,'opacity':.75},  # debounce
                               {'time_s':1.42,'trial_id':1,'opacity':.78}])
    assert result[0]['response_code'] == 4
    assert result[0]['response_s'] == 1.36 and result[1]['response_s'] == 1.42
    assert result[0]['rt_s'] > .8  # belongs to previous image, not whichever is visible
    assert score_trials(rows,[{'time_s':1.0,'trial_id':1,'opacity':.25}])[0]['rt_s'] == 1.0


def behavior(n=120):
    return [{'category':'city','correct':True,'rt_s':.5 + .1*np.sin(i/7), 'duration_s':.8,
             'onset_s':i*.8} for i in range(n)]


def test_vtc_missing_degenerate_incomplete_and_ties():
    config=QualityConfig(min_valid_responses=10)
    t=behavior()
    v, labels, status=compute_vtc(t,config)
    assert status == 'ok' and set(labels) == {'stable_response','variable_response'}
    t[30]['rt_s']=None; t[30]['correct']=False
    v,labels,status=compute_vtc(t,config)
    assert np.isfinite(v[30])
    for i in range(40,65): t[i]['rt_s']=None; t[i]['correct']=False
    v,labels,status=compute_vtc(t,config)
    assert np.isnan(v[50]) and labels[50]=='unknown'
    assert set(compute_vtc(t,config,False)[1])=={'unknown'}
    assert set(compute_vtc(behavior(3),config)[1])=={'unknown'}
    constant=[dict(r,rt_s=.5) for r in behavior()]
    assert set(compute_vtc(constant,config)[1])=={'unknown'}


def test_duration_weighted_window_labels_require_75_percent():
    trials=[{'onset_s':0,'duration_s':3,'behavior_label':'stable_response'},
            {'onset_s':3,'duration_s':1,'behavior_label':'variable_response'}]
    assert window_label(trials,0,4)=='stable_response'
    trials[0]['duration_s']=2.99
    assert window_label(trials,0,4)=='unknown'
    trials[0]['timing_bad']=True
    assert window_label(trials,0,4)=='unknown'


def test_provisional_quality_flags_each_failure():
    fs=250; rng=np.random.default_rng(1); raw=rng.normal(0,3,(2500,2))+10000
    assert assess_quality(raw,fs,np.arange(len(raw)))['valid']
    for name, x in [('constant_signal',np.ones_like(raw)), ('nonfinite',raw*np.nan),
                    ('amplifier_saturation',raw+800000), ('large_filtered_swing',raw*1000)]:
        assert any(name in r for r in assess_quality(x,fs)['reasons'])
    pack=np.arange(len(raw));pack[500:]+=1
    assert 'packet_gap_or_reset' in assess_quality(raw,fs,pack)['reasons']
    assert not assess_quality(raw[:10],fs)['valid']


def test_packet_timing_interpolation_and_validation():
    packets=[{'sample_start':0,'sample_count':4,'host_monotonic_s':10.03},
             {'sample_start':4,'sample_count':4,'host_monotonic_s':10.07}]
    np.testing.assert_allclose(sample_times(packets,8,100),10+np.arange(8)/100)
    with pytest.raises(ValueError): sample_times(packets,9,100)
    packets[-1]['host_monotonic_s']=9
    with pytest.raises(ValueError): sample_times(packets,8,100)


@pytest.fixture
def saved_session(tmp_path):
    directory=tmp_path/'session';directory.mkdir()
    config=QualityConfig(min_valid_responses=4)
    sequence=make_sequence(7,2,20)
    start=1000.0
    events=[]
    for i,t in enumerate(sequence):
        t['scheduled_onset_s']=start+i*.8
        events.append({'event':'trial_onset','trial_id':i,'onset_s':t['scheduled_onset_s']})
        if t['category']=='city':
            rt=.5+.1*np.sin(i)
            events.append({'event':'press','time_s':t['scheduled_onset_s']+rt,'trial_id':i,
                           'opacity':rt/.8,'phase':t['phase']})
    duration=len(sequence)*.8
    fs=100; n=round(duration*fs)
    rec=Recorder(directory,['Ch1','Ch2'],fs,meta={'device':{'serial':'SECRET','address':'BLE-ID'}},packet_timing=True)
    rec.start()
    rec.push(np.arange(n),np.zeros(n),np.random.default_rng(3).normal(0,4,(n,2)),0,start+(n-1)/fs)
    rec.stop()
    meta={'state':'complete','simulated':False,'quality_config':config.to_dict(),'eeg_file':rec.path.name,
          'packet_timing_file':rec.packet_path.name,'fs_hz':fs,'gain':3,'notch_hz':None,
          'ended_monotonic_s':start+duration,'device':{'serial':'SECRET','address':'BLE-ID'}}
    write_json(directory/'session.json',meta);write_json(directory/'sequence.json',sequence)
    (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    return directory


def test_complete_processing_reproducible_and_export_scrubs_identifiers(saved_session,tmp_path):
    directory=saved_session
    summary=process_session(directory)
    assert summary['accuracy']==1 and summary['training_eligible_windows']>0
    before=(directory/'windows.csv').read_bytes()
    assert summary==process_session(directory)
    assert before==(directory/'windows.csv').read_bytes()
    path=export_session(directory,tmp_path/'session.zip')
    with zipfile.ZipFile(path) as z:
        assert 'windows.csv' in z.namelist()
        assert all('SECRET' not in z.read(n).decode() and 'BLE-ID' not in z.read(n).decode()
                   for n in z.namelist() if n.endswith('.json'))
    assert 'SECRET' in (directory/'session.json').read_text()  # raw metadata unchanged
    with pytest.raises(FileExistsError): export_session(directory,path)


def test_partial_run_never_gets_behavior_labels(saved_session):
    path=saved_session/'session.json'; meta=json.loads(path.read_text());meta['state']='aborted';write_json(path,meta)
    summary=process_session(saved_session)
    assert summary['training_eligible_windows']==0
    assert summary['window_counts']['stable_response']==0
    assert summary['window_counts']['variable_response']==0


def test_timing_and_eeg_gaps_exclude_windows(saved_session):
    events=saved_session/'events.jsonl'
    with events.open('a') as f: f.write(json.dumps({'event':'timing_failure','trial_ids':list(range(24))})+'\n')
    summary=process_session(saved_session)
    assert summary['training_eligible_windows']==0
    assert 'presentation_timing_failure' in (saved_session/'windows.csv').read_text()


def test_export_rejects_path_traversal(saved_session,tmp_path):
    path=saved_session/'session.json';meta=json.loads(path.read_text());meta['eeg_file']='../secret.csv';write_json(path,meta)
    with pytest.raises(ValueError): export_session(saved_session,tmp_path/'bad.zip')


def test_export_preserves_existing_temporary_file(saved_session,tmp_path):
    process_session(saved_session)
    existing=tmp_path/'export.zip.tmp';existing.write_text('do not overwrite')
    with pytest.raises(FileExistsError): export_session(saved_session,tmp_path/'export.zip')
    assert existing.read_text()=='do not overwrite'


def test_unknown_and_simulation_have_explicit_exclusion_reasons(saved_session):
    path=saved_session/'session.json';meta=json.loads(path.read_text());meta.update(state='aborted',simulated=True)
    write_json(path,meta);process_session(saved_session)
    windows=(saved_session/'windows.csv').read_text()
    for reason in ('simulated_data','unknown_behavior','incomplete_session'): assert reason in windows


@pytest.mark.parametrize('settings', [{'frame_gap_s':0}, {'preview_s':.5}, {'min_valid_responses':2},
                                     {'max_interpolation_gap_trials':1.5}, {'rail_fraction':float('nan')}])
def test_quality_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError): QualityConfig(**settings)
