import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import matplotlib.pyplot as plt
import seaborn as sns

logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('UNIFORM FIT for EPISODE DURATION')

    episode_filename = os.path.join(
        args.confidential_dir, "cleaned_data.csv")
    # id,interarrival_days
    df = pd.read_csv(episode_filename)
    grouped = df.groupby('Unique Subject Identifier')

    df_unif = pd.DataFrame(columns=['id', 'episode_duration_mean',
                                    'episode_duration_var', 'episode_duration_count',
                                    'uniform_upper_bound', 'uniform_var'],
                           index=np.arange(len(grouped)))

    for i, (subject, df_subject) in enumerate(grouped):
        episode_duration = []
        # Can have a VOC and an ACS on the same day, but it's really the
        # same episode
        for date, df_date in df_subject.groupby('start_epoch_coalesced'):
            episode_duration.append(
                df_date['episode_duration_days_coalesced'].values[0])
            if len(df_date) > 1:
                # If VOC and ACS together, duration should be equal.
                assert np.var(
                    df_date['episode_duration_days_coalesced'].values) == 0.0
        episode_duration = np.array(episode_duration)
        # Uniform distribution across [1,B]
        max_observed = np.max(episode_duration)
        num_observed = 1.0 * len(episode_duration)
        # Minimum-variance unbiased estimator for uniform distribution:
        B = np.round(max_observed *
                     (1.0 + 1.0 / num_observed) - 1)
        B_var = np.square(B - 1) / 12
        results = {'id': subject, 'episode_duration_mean': np.mean(episode_duration),
                   'episode_duration_var': np.var(episode_duration),
                   'episode_duration_count': len(episode_duration),
                   'uniform_upper_bound': B, 'uniform_var': B_var}
        for r in results:
            df_unif[r].values[i] = results[r]

    df_unif.to_csv(os.path.join(
        args.working_dir, 'params_uniform_episode_duration.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
