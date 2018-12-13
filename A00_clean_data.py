# Original data is stored as an Excel spreadsheet.
# Add new columns to flag data anomalies and add some
# utility columns; store output as CSV file.  Read and
# write to "confidential_data/" directory.

import numpy as np
import pandas as pd
import os
import time
import utils_sickle_stats as utils
logger = utils.logger


# Convert string "2014-11-02" to epoch time
def to_epoch(s):
    t = time.strptime(s, "%Y-%m-%d")
    return(time.mktime(t))


def main(args):

    logger.info('==================================')
    logger.info('DATA CLEANING')

    input_filename = os.path.join(args.confidential_dir, args.input_file)

    # [u'Unique Subject Identifier', u'Informed consent date',
    #  u'Infusion Date', u'Type of VOE', u'Onset date', u'Resolution date',
    #  u'No. of VOEs']
    df = pd.read_excel(input_filename).drop_duplicates()

    # We will leave the original columns unchanged, and only add derived
    # columns

    # Distinguish data from earlier, less-effective therapy.  Will be false if
    # therapy occured too long ago, or if no therapy was applied.
    df['patient_treated'] = False
    for i in range(len(df)):
        df['patient_treated'].values[i] = (
            (int(df['Unique Subject Identifier'].values[i][-4:]) >= 1315) and
            (pd.notnull(df['Infusion Date'].values[i])))

    # Standardize dates for episode onset/resolution into epoch seconds, and record
    # duration of event in days.  Note that if, e.g., onset date = resolution date, then
    # this episode lasts 1 day, not zero days.
    #
    # If a field is missing, value is set to NaN.

    df['start_epoch'] = 0.0
    df['end_epoch'] = 0.0
    df['episode_duration_days'] = 0.0

    for i in range(len(df)):
        start_date = df['Onset date'].values[i]
        end_date = df['Resolution date'].values[i]

        # A well-formatted date looks like '2018-01-15';
        # partial dates (like just '2018-01' or '2018') exist.
        for col, d in [('start_epoch', start_date), ('end_epoch', end_date)]:
            if len(d.split('-')) == 3:
                df[col].values[i] = to_epoch(d)
            else:
                df[col].values[i] = np.NaN
        df['episode_duration_days'].values[i] = 1 + (
            df['end_epoch'].values[i] - df['start_epoch'].values[i]) / 86400

    # Handle missing dates by coalescing from other date (e.g., if
    # start date of episode is present but end date is missing, insert end date
    # into start date)
    df['start_epoch_coalesced'] = df['start_epoch']
    df['end_epoch_coalesced'] = df['end_epoch']
    df['episode_duration_days_coalesced'] = df['episode_duration_days']

    time_list = ['start', 'end']
    for i in range(2):
        s = time_list[i]
        t = time_list[1 - i]
        df['%s_epoch_coalesced' % s] = df['%s_epoch' % s]
        invalid = pd.isnull(df['%s_epoch_coalesced' % s])
        df['%s_epoch_coalesced' % s].values[
            invalid] = df['%s_epoch' % t].values[invalid]
    invalid = pd.isnull(df['episode_duration_days_coalesced'])
    df['episode_duration_days_coalesced'].values[invalid] = 1

    # Can have an ACS and VOC on the same day; we add a column so we can select
    # a unique representative of each event if we want to.
    grouped_date = df.groupby(['Unique Subject Identifier', 'start_epoch_coalesced'])
    df['unique_event_witness'] = True
    for (subj, start_epoch), df_date in grouped_date:
        if len(df_date) == 1:
            continue
        indices = grouped_date.groups[subj, start_epoch]
        for ind in indices[1:]:
            df['unique_event_witness'].values[ind] = False

    clean_filename = os.path.join(
        args.confidential_dir, args.clean_file)
    logger.info("Writing file %s" % clean_filename)
    df.to_csv(clean_filename, index=False)

    # Extract interarrival times
    grouped = df.groupby('Unique Subject Identifier')
    df_interarrival = pd.DataFrame(
        columns=['id', 'interarrival_days', 'episode_duration_days'])
    for subject, df_subject in grouped:
        df_subject = df_subject.sort_values(by=['start_epoch_coalesced'])
        onsets = df_subject['start_epoch_coalesced'].values / 86400
        durations = 1.0 + (df_subject['end_epoch_coalesced'].values -
                           df_subject['start_epoch_coalesced'].values) / 86400
        assert np.all(np.isfinite(onsets))
        # onsets = onsets[np.isfinite(onsets)]
        interarrival_times = np.diff(onsets)
        # Note: it's possible to have a VOC and ACS at the same time;
        # we should coalesce these into a single episode, per below.
        distinct_indices = interarrival_times > 0
        interarrival_times = interarrival_times[distinct_indices]
        durations = durations[:-1][distinct_indices]
        # There's a little weirdness because of daylight savings time; round to nearest day
        interarrival_times = np.round(interarrival_times).astype(int)
        durations = np.round(durations).astype(int)
        for t, d in zip(interarrival_times, durations):
            df_interarrival = df_interarrival.append(
                {'id': subject, 'interarrival_days': t, 'episode_duration_days': d}, ignore_index=True)

    interarrival_filename = os.path.join(
        args.confidential_dir, "interarrival_times.csv")
    logger.info("Writing file %s" % interarrival_filename)
    df_interarrival.to_csv(interarrival_filename, index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
