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

    # Unique Subject Identifier,Informed consent date,Infusion Date,Type of VOE,
    # Onset date,Resolution date,No. of VOEs,patient_treated,start_epoch,end_epoch,
    # episode_duration_days,start_epoch_coalesced,end_epoch_coalesced,
    # episode_duration_days_coalesced
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
            if True:
                # Interarrival time between episodes
                plt.figure(figsize=(8, 6))
                interarrival_times = df_subject['interarrival_days'].values
                sns.distplot(interarrival_times, rug=True)

                time_span_days = subject_to_timespan[subject]
                plt.title('Subject %s; N=%d episodes over %d days ($\mu$=%d, $\sigma$=%.1f, $\sigma^2$=%d)' % (
                    subject[-4:], len(df_subject) + 1,
                    time_span_days, np.mean(interarrival_times),
                    np.std(interarrival_times),
                    np.var(interarrival_times)))
                plt.xlabel('Episode interarrival time in days')

            if True:
                # Duration of episodes
                plt.figure(figsize=(8, 6))
                df_m = df.iloc[grouped.groups[subject]]
                episode_duration = []
                # Can have a VOC and an ACS on the same day, but it's really the
                # same episode
                for date, df_date in df_m.groupby('start_epoch_coalesced'):
                    episode_duration.append(df_date['episode_duration_days_coalesced'].values[0])
                    if len(df_date)>1:
                        # If VOC and ACS together, duration should be equal.
                        assert np.var(df_date['episode_duration_days_coalesced'].values) == 0.0
                episode_duration = np.array(episode_duration)

                dither_width = 0.5
                episode_dither = dither_width * 2.0 * (np.random.rand(len(episode_duration)) - 0.5)
                sns.distplot(episode_duration + episode_dither, rug=True)

                time_span_days = subject_to_timespan[subject]
                plt.title('Subject %s; Episode duration; N=%d episodes over %d days ($\mu$=%d, $\sigma$=%.1f, $\sigma^2$=%d)' % (
                    subject[-4:], len(episode_duration),
                    time_span_days, np.mean(episode_duration),
                    np.std(episode_duration),
                    np.var(episode_duration)))
                plt.xlabel('Episode duration in days')


            plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    preliminary_stats(args)
