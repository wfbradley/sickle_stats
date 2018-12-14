import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from version_sickle_stats import __version__


# Create directory only if needed.  (Makes nested directories if needed,
# e.g. "new1/new2/new3/".)
def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


logger = logging.getLogger('Sickle Cell Stats logger')
logger_initialized = False


def filter_cleaned_data(df, uniquify_events=False):
    if uniquify_events:
        df = df.iloc[df['unique_event_witness'].values].reset_index(drop=True)

    return(df)


# A Python generator that partitions events across time.
#
# This generator iterates through each subject.  It groups data
# by subject.  For each subject, it partitions time into
# (non-overlapping) windows of num_days days, and counts how
# many days the subject was hospitalized during each window.
# The generator then returns (subject, vector of counts).
#
# We offer three ways of counting:
#  onset:  Count episodes, at time of episode onset
#  hospital: Count days in hospital during the window
#         (thus, the maximum count = num_days)
#  hospital_onset: Count days in hospital, but credit
#       all hospital days at time on onset (so count
#       is unbounded)
def event_partition_generator(args, df=None, num_days=7,
                              count_type='hospital_onset'):
    assert count_type in ['onset', 'hospital', 'hospital_onset']

    if df is None:
        filename = os.path.join(
            args.confidential_dir, args.clean_file)
        df = pd.read_csv(filename)
        df = filter_cleaned_data(df, uniquify_events=True)

    grouped = df.groupby('Unique Subject Identifier')
    window_seconds = 86400 * num_days
    for subj, df_subj in grouped:
        df_subj = df_subj.sort_values(by=['start_epoch_coalesced'])
        earliest_epoch = np.min(df_subj.start_epoch_coalesced.values)
        # "consent_epoch" should be constant, so we arbitrarily take the first one
        latest_epoch = df_subj['consent_epoch'].values[0]
        num_windows = int(np.floor(
            (latest_epoch - earliest_epoch) / window_seconds))
        if num_windows == 0:
            logger.info("No windows for Subject %s" % (subj[-4:]))
            continue
        counts = np.zeros(num_windows)

        for i in range(len(df_subj)):
            duration_days = int(df_subj[
                'episode_duration_days_coalesced'].values[i])
            for j in range(duration_days):
                w = int((df_subj['start_epoch_coalesced'].values[i] - 
                        earliest_epoch + j * 86400) / window_seconds)
                if count_type == 'onset':
                    counts[w] += 1
                    break
                elif count_type == 'hospital_onset':
                    counts[w] += duration_days
                    break
                elif count_type == 'hospital':
                    counts[w] += 1
                    # do *not* break!
        yield (subj, counts)


def check_permissions(args):
    if os.getuid() == 0:
        logger.info('Running as root.')
        return
    # If here, we are not root.
    if args.halt:
        logger.info('Not running as root.')
        raise Exception
    return


def initialize_logger(args):
    global logger_initialized

    if logger_initialized:
        return
    logger_initialized = True

    safe_mkdir(args.working_dir)
    log_file = os.path.join(args.working_dir, 'run.log')

    if not (args.keep_old_logs):
        try:
            os.remove(log_file)
        except Exception:
            pass
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("##############  Beginning logging  ##############")
    logger.info("Sickle Stats version %s" % __version__)
    logger.info("Command line: %s" % (' '.join(sys.argv)))
    logger.info("Runtime parameters:")
    for f in args.__dict__:
        if f.startswith('__'):
            continue
        logger.info("   %s  :  %s" % (f, args.__dict__[f]))


def parse_arguments():

    parser = argparse.ArgumentParser(description='Supervised ML classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--confidential_dir', type=str, default='confidential_data',
                        help='Directory for reading and writing confidential data.')
    parser.add_argument('--working_dir', type=str, default='data',
                        help='Directory for all working files and output')
    parser.add_argument('--input_file', type=str, default='l_voc_acs_mh.xlsx',
                        help='Input Excel data file within confidential_data_dir')
    parser.add_argument('--clean_file', type=str, default='cleaned_data.csv',
                        help='Cleaned data file (csv) within confidential_data_dir')
    parser.add_argument('--daily_file', type=str, default='daily_file.csv',
                        help='Cleaned data file converted to daily format (csv) within confidential_data_dir')
    parser.add_argument('--draw_plots', action='store_true', default=False,
                        help='Render plots to screen')

    parser.add_argument('--keep_old_logs', action='store_true', default=False,
                        help='By default, old log file is deleted; this preserves it.')
    args = parser.parse_args()

    # Fully qualify "~"s from path names
    args.confidential_dir = os.path.expanduser(args.confidential_dir)
    args.working_dir = os.path.expanduser(args.working_dir)

    return(args)
