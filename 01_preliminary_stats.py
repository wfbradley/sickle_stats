import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import utils_sickle_stats as utils
import seaborn as sns

logger = utils.logger
sns.set()


def preliminary_stats(args):

    logger.info('==================================')
    logger.info('PRELIMINARY STATS')

    filename = os.path.join(
        args.confidential_dir, args.clean_file)
    logger.info("Summarizing data from %s" % filename)
    # Unique Subject Identifier,Informed consent date,Infusion Date,
    # Type of VOE,Onset date,Resolution date,No. of VOEs,
    # patient_treated,start_epoch,end_epoch,episode_duration_days
    df = pd.read_csv(filename)

    interarrival_filename = os.path.join(
        args.confidential_dir, "interarrival_times.csv")
    # id,interarrival_days
    df_interarrival = pd.read_csv(interarrival_filename)

    grouped = df.groupby('Unique Subject Identifier')
    # subject_list = grouped.groups.keys()
    logger.info("Number of subjects: %d" % len(grouped))
    logger.info("  Subj:  N VOEs; first; last; consent")
    subject_to_timespan = {}
    for subject, df_subject in grouped:
        logger.info("  %s:  %d ; %s ; %s ; %s" % (
            subject[-4:], len(df_subject),
            np.nanmin(df_subject['Onset date'].values),
            np.nanmax(df_subject['Onset date'].values),
            df_subject['Informed consent date'].values[0],
        ))
        time_span_days = (np.nanmax(df_subject['start_epoch'].values) -
                          np.nanmin(df_subject['start_epoch'].values)) / 86400
        subject_to_timespan[subject] = time_span_days

    grouped_interarrival = df_interarrival.groupby('id')
    for subject, df_subject in grouped_interarrival:
        if args.draw_plots:
            plt.figure(figsize=(8, 6))
            interarrival_times = df_subject['interarrival_days'].values
            sns.distplot(interarrival_times, rug=True)

            time_span_days = subject_to_timespan[subject]
            plt.title('Subject %s; N=%d episodes over %d days (%d days/eps)' % (
                subject[-4:], len(df_subject) + 1,
                time_span_days, 1.0 * time_span_days / len(df_subject)))
            plt.xlabel('Episode interarrival time in days')

            plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    preliminary_stats(args)
