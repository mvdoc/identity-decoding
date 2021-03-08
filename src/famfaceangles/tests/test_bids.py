"""Module containing tests for famfaceangles.bids"""

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from os.path import dirname, abspath, join
from ..bids import make_events


def _fake_face_stim(onset, stim_file, stim_size,
                    stim_poss, reaction_time='', response=''):
    """
    Code to emulate stimulus presentation

    Parameters
    ----------
    onset
    stim_file
    reaction_time
    response
    stim_size
    stim_poss

    Returns
    -------
    """

    lines = []
    stim_template = \
        '{onset:.4f}\tEXP\t{stim_file}: {prop} = array([{x}, {y}])\n'

    stim_code = 5

    lines.append('{0}\tEXP\tSTIM: {1}\n'.format(onset, stim_code))
    lines.append(
        stim_template.format(
            onset=onset,
            stim_file=stim_file,
            prop='size',
            x=stim_size[0],
            y=stim_size[1]
        )
    )

    onset_ = onset
    for i, pos in enumerate(stim_poss):
        lines.append(
            stim_template.format(
                onset=onset_,
                stim_file=stim_file,
                prop='pos',
                x=pos[0], y=pos[1]
            )
        )
        onset_ += 0.62

    if response and reaction_time:
        lines.append(
            '{0}\tDATA\tRESPONSE: {1}\n'.format(onset + reaction_time,
                                                response)
        )

    return lines


def test_make_events():
    testlog = join(dirname(abspath(__file__)), 'testlog.txt')
    with open(testlog, 'r') as f:
        lines = f.readlines()

    df = make_events(lines)
    assert(len(df) > 0)

    response2lbl = {'1': 'same', '2': 'different'}
    onsets = [3.5, 8.5, 13.5, 19.5]
    identities = ['id1', 'id3', 'id2', 'id2']
    orientations = ['a4', 'a3', 'a5', 'a1']
    stim_files = [
        'stim/familiar/{0}_{1}.png'.format(i, o)
        for i, o in zip(identities, orientations)
    ]

    sizes = [(500, 500) for _ in onsets]
    poss = [2 + np.random.randn(2)*10 for _ in onsets]
    reaction_times = [2.3, 1.2, '', 3.4]
    responses = ['1', '1', '', '2']

    lines = []
    for on, sf, sz, rt, res in zip(onsets, stim_files, sizes,
                                   reaction_times, responses):
        lines.extend(
            _fake_face_stim(on, sf, sz, poss,
                            reaction_time=rt, response=res))

    df = make_events(lines)

    assert(len(df)) == len(onsets)
    assert_array_equal(df.onset, onsets)
    assert_array_equal(df.identity, identities)
    assert_array_equal(df.orientation, orientations)
    assert_array_equal(df.stim_file, stim_files)
    assert_array_equal(df.duration,
                       np.round(np.diff(onsets), 0).tolist() + [5])
    assert_array_equal([np.round(r, 2) if isinstance(r, float)
                        else r for r in df.reaction_time],
                       np.array(reaction_times, dtype=str))
    assert_array_equal(df.response, [response2lbl.get(r, '')
                                     for r in responses])
    assert(df.stim_type.unique() == 'face')

    # add fixation at the end
    lines.extend(
        ['{0}\tEXP\tSTIM: 0'.format(onsets[-1] + 5.)]
    )
    df_ = make_events(lines)
    assert(df.to_dict() == df_.iloc[:-1].to_dict())
    assert(df_.iloc[-1].stim_type == 'fixation')


def test_make_events_startfixation():
    """Check what happens when the run starts with fixation"""

    lines = """15.8360         EXP     START OF EXPERIMENT
    30.8631         EXP     STIM: 0
    30.8881         EXP     0
    35.8652         EXP     STIM: 0
    35.8883         EXP     0
    40.8662         EXP     STIM: 10
    40.9018         EXP     stim/stranger/id2_a5.png: pos = array([ 3.56839343,  9.24224815])
    40.9018         EXP     stim/stranger/id2_a5.png: size = array([ 532.10222004,  532.10222004])
    41.5018         EXP     stim/stranger/id2_a5.png: pos = array([ 7.9594968 ,  0.60281356])
    42.1019         EXP     stim/stranger/id2_a5.png: pos = array([ 7.75116244,  0.74977234])"""

    df = make_events(lines.split('\n'))
    assert(len(df) == 3)


def test_make_events_tworesponses():
    """Check that we manage correctly when we have two responses"""
    exp_start = 15.8360
    stim_on = 40.9018
    resps = [43.0111, 43.4111, 43.6111]
    lines = """{0:.4f}         EXP     START OF EXPERIMENT
    30.8631         EXP     STIM: 0
    30.8881         EXP     0
    35.8652         EXP     STIM: 0
    35.8883         EXP     0
    40.8662         EXP     STIM: 10
    {1:.4f}         EXP     stim/stranger/id2_a5.png: pos = array([ 3.56839343,  9.24224815])
    40.9018         EXP     stim/stranger/id2_a5.png: size = array([ 532.10222004,  532.10222004])
    41.5018         EXP     stim/stranger/id2_a5.png: pos = array([ 7.9594968 ,  0.60281356])
    42.1019         EXP     stim/stranger/id2_a5.png: pos = array([ 7.75116244,  0.74977234])
    {2:.4f}        DATA    RESPONSE: 2
    {3:.4f}        DATA    RESPONSE: 1
    {4:.4f}        DATA    RESPONSE: 2""".format(exp_start, stim_on, *resps)

    df = make_events(lines.split('\n'))
    assert(len(df) == 3)
    face = df[df.stim_type == 'face']
    assert np.round(stim_on - exp_start, 2) == float(face.onset)
    assert np.round(resps[-1] - exp_start, 2) == float(face.onset) + \
                                              float(face.reaction_time)
