import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import nbinom_fit
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('COUNT NEGATIVE BINOMIAL FIT')

    df_nbinom_params = pd.DataFrame(columns=['id', 'r', 'prob', 'count_mean',
                                             'count_var', 'count_len', 'loglikelihood'])

    for i, (subj, counts) in enumerate(utils.event_partition_generator(args, num_days=7)):
        df_nbinom_params.loc[i] = [
            np.nan for j in range(len(df_nbinom_params.columns))]

        df_nbinom_params['id'].values[i] = subj
        df_nbinom_params['count_mean'].values[i] = np.mean(counts)
        df_nbinom_params['count_var'].values[i] = np.var(counts)
        df_nbinom_params['count_len'].values[i] = len(counts)

        try:
            nbinom_params = nbinom_fit.nbinom_fit(counts)
            # I believe the "size" argument is actually the parameter "r".
            (r, prob) = (nbinom_params['size'], nbinom_params['prob'])
            loglikelihood = nbinom_fit.log_likelihood((r, prob), counts)
        except Exception:
            logger.info('Could not fit negative binomial on %s' % subj)
            (r, prob, loglikelihood) = (np.nan, np.nan, np.nan)

        df_nbinom_params['r'].values[i] = r
        df_nbinom_params['prob'].values[i] = prob
        df_nbinom_params['loglikelihood'].values[i] = loglikelihood

    df_nbinom_params.to_csv(os.path.join(
        args.working_dir, 'params_nbinom_count.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
