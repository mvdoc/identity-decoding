"""Module containing code to aid BIDSification"""
import numpy as np
import pandas as pd
import re


def _store(datastore, **kwargs):
    """Store data into a dictionary, passed as kwargs. Keys must be existing in
    datastore in order to be added. If a key is not passed, an empty string
    is added in order to have the same length. Operates in place.

    Parameters
    ----------
    datastore : dict
        values are lists
    """
    for key in datastore:
        if key in kwargs:
            datastore[key].append(kwargs[key])
        else:
            datastore[key].append('')


def _1stpass_log(lines):
    """First pass of the logfiles. Store everything into a dictionary for
    later use"""

    stim_match = re.compile(
        r'(stim\/.*\/.*\.png).*(pos|size).*array\(\[(.*),(.*)\]\)')

    columns = [
        'onset',
        'stim_type',
        'stim_file',
        'stim_pos_x',
        'stim_pos_y',
        'stim_width',
        'stim_height',
        'identity',
        'orientation',
        'response'
    ]

    datastore = {c: [] for c in columns}

    start_time = 0.
    for line in lines:
        parts = line.split()
        start_stim = round(float(parts[0]) - start_time, 2)
        if 'START OF EXPERIMENT' in line:
            start_time = float(parts[0])
        # check if we have a fixation cross
        if 'STIM' in line:
            # starting stimulus presentation, check which stim it is
            stim_code = int(parts[-1])
            if stim_code == 0:
                stim_type = 'fixation'
                _store(datastore,
                       stim_type=stim_type,
                       onset=start_stim)
        # check if we have a face stimulus
        if 'Created' not in line and 'stim' in line and 'png' in line:
            stim_type = 'face'
            file_name, prop, x, y = map(
                lambda s: s.strip(),
                stim_match.search(line).groups())
            identity, orientation = \
                file_name.replace('.png', '').split('/')[-1].split('_')
            x = int(round(float(x)))
            y = int(round(float(y)))

            if prop == 'size':
                stim_width = x
                stim_height = y
                stim_pos_x = ''
                stim_pos_y = ''
            else:
                stim_pos_x = x
                stim_pos_y = y
                stim_width = ''
                stim_height = ''

            _store(datastore,
                   stim_type=stim_type,
                   onset=start_stim,
                   stim_file=file_name,
                   identity=identity,
                   orientation=orientation,
                   stim_width=stim_width,
                   stim_height=stim_height,
                   stim_pos_x=stim_pos_x,
                   stim_pos_y=stim_pos_y)
        # check if we have a button press to store
        if 'DATA' in line:
            stim_type = 'button press'
            resp = line.split()[-1]
            response = 'same' if resp == '1' else 'different'
            _store(datastore,
                   stim_type=stim_type,
                   onset=start_stim,
                   response=response)
    return pd.DataFrame(datastore, columns=columns)


def _2ndpass_log(df):
    """Performs second pass of logfiles -- extends size and pos to all
    matching trials

    Parameters
    ----------
    df : pd.DataFrame
        obtained from _1stpass_log

    Returns
    -------
    df : pd.DataFrame
    """
    columns = [
        'onset',
        'duration',
        'stim_type',
        'identity',
        'orientation',
        'stim_file',
        'stim_pos_x',
        'stim_pos_y',
        'stim_width',
        'stim_height',
        'response'
    ]

    events = {c: [] for c in columns}

    nrows = len(df)
    irow = 1
    prev_trial = df.iloc[0].to_dict()
    # store this info
    _store(events, **prev_trial)

    while irow < nrows:
        trial = df.iloc[irow].to_dict()
        stim_type = trial['stim_type']
        time_diff = trial['onset'] - prev_trial['onset']
        # extend the properties (size and pos) to all trials
        if stim_type == 'face' and time_diff < 1.0:
            if not trial['stim_width']:
                trial['stim_width'] = prev_trial['stim_width']
                trial['stim_height'] = prev_trial['stim_height']
            if not trial['stim_pos_x']:
                trial['stim_pos_x'] = prev_trial['stim_pos_x']
                trial['stim_pos_y'] = prev_trial['stim_pos_y']
        _store(events, **trial)
        prev_trial = trial
        irow += 1
    return pd.DataFrame(events, columns=columns)


def _finalpass_log(df):
    """

    Parameters
    ----------
    df : pd.DataFrame
        dataframe obtained from _2ndpass_log

    Returns
    -------
    df : pd.DataFrame
        dataframe formatted according to BIDS v1.0.2
    """
    columns = [
        'onset',
        'duration',
        'stim_type',
        'identity',
        'orientation',
        'stim_file',
        'reaction_time',
        'response',
    ]

    events = {c: [] for c in columns}

    # Let's make it in a more usable format
    prev_trial = df.iloc[0].to_dict()
    _store(events,
           **{k: prev_trial[k]
              for k in ['onset', 'duration', 'stim_type',
                        'identity', 'orientation', 'stim_file',
                        'response']}
           )

    for irow in range(1, len(df)):
        trial = df.iloc[irow].to_dict()

        # some heuristics to figure out if it's a new trial
        is_new_trial = \
            (trial['identity'], trial['orientation']) != \
            (prev_trial['identity'], prev_trial['orientation']) \
            or (trial['stim_width'] != prev_trial['stim_width']) \
            or trial['onset'] - events['onset'][-1] > 4.0

        # store reaction time and response for button press
        if trial['stim_type'] == 'button press':
            events['reaction_time'][-1] = \
                str(np.round(trial['onset'] - events['onset'][-1], 2))
            events['response'][-1] = trial['response']
        elif trial['stim_type'] == 'fixation' or is_new_trial:
            _store(events,
                   **{k: trial[k]
                      for k in ['onset', 'duration', 'stim_type',
                                'identity', 'orientation', 'stim_file']})
        prev_trial = trial

    df_events = pd.DataFrame(events, columns=columns)
    # add duration
    df_events.iloc[:-1, 1] = np.round(np.diff(df_events.onset), 0)
    df_events.iloc[-1, 1] = 5.
    # remove remaining trials with 0 duration
    df_events = df_events[df_events.duration != 0].reset_index()
    return df_events[columns]


def make_events(lines):
    df = _1stpass_log(lines)
    df = _2ndpass_log(df)
    return _finalpass_log(df)
