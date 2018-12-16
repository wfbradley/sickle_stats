import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import nbinom_fit
import seaborn as sns
import statsmodels.api as sm

logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('COUNT POISSON FIT')

    df_poisson_params = pd.DataFrame(columns=['id', 'lambda', 'count_mean',
                                              'count_var', 'count_len', 'loglikelihood'])

    for i, (subj, counts) in enumerate(utils.event_partition_generator(args, num_days=7)):
        df_poisson_params.loc[i] = [
            np.nan for j in range(len(df_poisson_params.columns))]

        df_poisson_params['id'].values[i] = subj
        df_poisson_params['count_mean'].values[i] = np.mean(counts)
        df_poisson_params['count_var'].values[i] = np.var(counts)
        df_poisson_params['count_len'].values[i] = len(counts)

        try:
            res = sm.Poisson(counts, np.ones_like(counts)).fit(disp=0)
            lambda_param = res.params[0]
            loglikelihood = -res.llf
        except Exception:
            logger.info('Could not fit negative binomial on %s' % subj)
            (lambda_param, loglikelihood) = (np.nan, np.nan)

        df_poisson_params['lambda'].values[i] = lambda_param
        df_poisson_params['loglikelihood'].values[i] = loglikelihood

    df_poisson_params.to_csv(os.path.join(
        args.working_dir, 'params_poisson_count.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
