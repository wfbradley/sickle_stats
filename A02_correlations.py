import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import utils_sickle_stats as utils
import seaborn as sns
from scipy import stats


def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2


logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('PRELIMINARY CORRELATIONS')

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
                interarrival_times = df_subject['interarrival_days'].values

                if len(interarrival_times) >= 4:
                    try:
                        jj = sns.jointplot(interarrival_times[:-1], interarrival_times[1:],
                                           kind="reg", stat_func=r2)
                        time_span_days = subject_to_timespan[subject]
                        jj.ax_marg_x.set_title(
                            'Subject %s; N=%d episodes over %d days ($\mu$=%d, $\sigma$=%.1f, $\sigma^2$=%d)' % (
                                subject[-4:], len(df_subject) + 1,
                                time_span_days, np.mean(interarrival_times),
                                np.std(interarrival_times),
                                np.var(interarrival_times)))
                        jj.ax_joint.set_xlabel('Earlier interarrival')
                        jj.ax_joint.set_ylabel('Later interarrival')
                        plt.tight_layout()
                    except Exception:
                        logger.info(
                            'Interarrival correlation plot failure on Subject %s' % subject)

            if True:
                # Duration of episodes
                df_m = df.iloc[grouped.groups[subject]]
                episode_duration = []
                # Can have a VOC and an ACS on the same day, but it's really the
                # same episode
                for date, df_date in df_m.groupby('start_epoch_coalesced'):
                    episode_duration.append(
                        df_date['episode_duration_days_coalesced'].values[0])
                    if len(df_date) > 1:
                        # If VOC and ACS together, duration should be equal.
                        assert np.var(
                            df_date['episode_duration_days_coalesced'].values) == 0.0
                episode_duration = np.array(episode_duration)
                if len(episode_duration) >= 4:
                    try:
                        jj = sns.jointplot(episode_duration[:-1], episode_duration[1:],
                                           kind="reg", stat_func=r2)
                        time_span_days = subject_to_timespan[subject]
                        jj.ax_marg_x.set_title(
                            'Subject %s; %d days in hospital over %d episodes ($\mu$=%.1f, $\sigma$=%.1f, $\sigma^2$=%d)' % (
                                subject[-4:], np.sum(episode_duration), len(
                                    episode_duration),
                                np.mean(episode_duration),
                                np.std(episode_duration),
                                np.var(episode_duration)))
                        jj.ax_joint.set_xlabel('Earlier duration')
                        jj.ax_joint.set_ylabel('Later duration')
                        plt.tight_layout()
                    except Exception:
                        logger.info(
                            'Episode duration plot failure on Subject %s' % subject)

            if True:
                # Duration versus interarrival time
                if len(df_subject) >= 10:
                    try:
                        jj = sns.jointplot('interarrival_days', 'episode_duration_days', data=df_subject,
                                           kind="reg", stat_func=r2)
                        time_span_days = subject_to_timespan[subject]
                        jj.ax_marg_x.set_title(
                            'Subject %s; Interarrival time vs duration; N=%d' % (
                                subject[-4:], len(df_subject)))
                        # jj.ax_joint.set_xlabel('Earlier duration')
                        # jj.ax_joint.set_ylabel('Later duration')
                        plt.tight_layout()
                        r2_value = r2(df_subject['interarrival_days'].values,
                                      df_subject['episode_duration_days'].values)
                        logger.info("Subj %s: R^2=%.2f, N=%d" %
                                    (subject[-4:], r2_value, len(df_subject)))
                    except Exception:
                        logger.info(
                            'Episode duration plot failure on Subject %s' % subject)

            plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
