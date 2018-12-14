import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import matplotlib.pyplot as plt
import tweedie_dist
import tweedie_fit
import seaborn as sns

logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('TWEEDIE COUNT FIT')

    df_tweedie_params = pd.DataFrame(columns=['id', 'mu', 'p', 'phi', 'count_mean',
                                             'count_var', 'count_len', 'loglikelihood'])

    for i, (subj, counts) in enumerate(utils.event_partition_generator(args, num_days=7)):
        df_tweedie_params.loc[i] = [
            np.nan for j in range(len(df_tweedie_params.columns))]

        df_tweedie_params['id'].values[i] = subj
        df_tweedie_params['count_mean'].values[i] = np.mean(counts)
        df_tweedie_params['count_var'].values[i] = np.var(counts)
        df_tweedie_params['count_len'].values[i] = len(counts)

        try:
            tweedie_params = tweedie_fit.tweedie_fit(counts)
            (mu, p, phi, loglikelihood) = (tweedie_params['mu'], tweedie_params['p'],
                                           tweedie_params['phi'], tweedie_params['loglikelihood'])
        except Exception:
            logger.info('Could not fit Tweedie on %s' % subj)
            (mu, p, phi, loglikelihood) = (np.nan, np.nan, np.nan, np.nan)

        df_tweedie_params['mu'].values[i] = mu
        df_tweedie_params['p'].values[i] = p
        df_tweedie_params['phi'].values[i] = phi
        df_tweedie_params['loglikelihood'].values[i] = loglikelihood

    df_tweedie_params.to_csv(os.path.join(
        args.working_dir, 'params_tweedie_count.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
