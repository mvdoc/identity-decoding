"""Tests for workflows"""
import numpy as np
import pandas as pd
import os.path as op
import pytest
import re
from six import iteritems
from ..workflows import (make_afni_stimtimes_exemplar, 
                         make_afni_stimtimes_localizer)

rx_run = re.compile('run([0-9]{2})')


def test_make_afni_stimtimes_exemplar(tmpdir):
    event_fn = op.abspath(op.join(op.dirname(__file__), 'events.tsv'))
    events = [event_fn,
              event_fn.replace('events', 'events1'),
              event_fn.replace('events', 'events2')]
    tmpdir.chdir()
    n_events = 3
    start = 2
    fns = make_afni_stimtimes_exemplar(events, start=start)
    assert(len(fns) == 22)  # 20 stims + first trial + response
    # check we have absolute paths
    for fn in fns:
        assert op.isabs(fn)
    # load onsets
    onsets = dict()
    for fn in fns:
        with open(fn, 'r') as f:
            onsets[fn] = f.readlines()
    # we should have n_events rows, and three repetitions for each stimulus
    for fn, onset in iteritems(onsets):
        if op.basename(fn) not in ['first_trial.txt', 'same_response.txt']:
            assert len(onset) == n_events + start
            for o in onset:
                if '*' not in o:
                    assert(len(o.split()) == 3)


@pytest.mark.parametrize("start", [0, 2, 5])
def test_make_afni_stimtimes_exemplar_perrun(tmpdir, start):
    event_fn = op.abspath(op.join(op.dirname(__file__), 'events.tsv'))
    events = [event_fn,
              event_fn.replace('events', 'events1'),
              event_fn.replace('events', 'events2')]
    tmpdir.chdir()
    n_events = len(events)
    fns = make_afni_stimtimes_exemplar(events, start=start, per_run=True)
    # 20 stims + first trial + response for each run
    assert(len(fns) == 22 * n_events)
    # check we have absolute paths
    for fn in fns:
        assert op.isabs(fn)
    # load onsets
    onsets = dict()
    for fn in fns:
        with open(fn, 'r') as f:
            onsets[op.basename(fn)] = f.readlines()
    # test length of onsets
    for fn, onset in iteritems(onsets):
        if not fn.startswith(('first_trial', 'same_response')):
            # we should have a variable number of rows depending on the run
            irun = int(rx_run.findall(fn)[0])
            assert len(onset) == irun + start
            for o in onset:
                if '*' not in o:
                    # we should have only three repetitions for each run
                    assert(len(o.split()) == 3)


def test_make_afni_stimtimes_exemplar_perrun_onsets(tmpdir):
    """Test the actual onset values"""
    event_fn = op.abspath(op.join(op.dirname(__file__), 'events.tsv'))
    events = [event_fn]
    tmpdir.chdir()
    n_events = len(events)
    fns = make_afni_stimtimes_exemplar(events, per_run=True)
    assert(len(fns) == 22 * n_events)
    # check we have absolute paths
    for fn in fns:
        assert op.isabs(fn)
    # load onsets
    onsets = dict()
    for fn in fns:
        with open(fn, 'r') as f:
            onsets[op.basename(fn)] = f.readlines()
    events_df = pd.read_csv(event_fn, sep='\t')
    # OK now let's test the actual onsets
    key_same = 'same_response_run01.txt'
    for trial in events_df.itertuples():
        if trial.Index == 0:
            key = 'first_trial_run01.txt'
            assert len(onsets[key][0].split()) == 1
            assert trial.onset == float(onsets[key][0])
        elif trial.stim_type == 'face':
            key = '{0}_{1}_run01.txt'.format(trial.identity, trial.orientation)
            assert '{0:.2f}'.format(trial.onset) in onsets[key][0].split()
            if trial.response == 'same':
                assert '{0:.2f}'.format(trial.onset + trial.reaction_time) in \
                       onsets[key_same][0].split()


def test_make_afni_stimtimes_localizer(tmpdir):
    event_fn = op.abspath(op.join(op.dirname(__file__), 
                                  'events_localizer.tsv'))
    tmpdir.chdir()
    n_events = 3
    stim_fns = make_afni_stimtimes_localizer(
        [event_fn for _ in range(n_events)]
    )
    # check we have absolute paths
    for fn in stim_fns:
        assert op.isabs(fn)
