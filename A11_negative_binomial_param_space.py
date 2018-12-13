import pandas as pd
import os
import utils_sickle_stats as utils
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = utils.logger
sns.set()


def main(args):

    logger.info('==================================')
    logger.info('NEGATIVE BINOMIAL PARAMETER SPACE VISUALIZATION')

    nbinom_filename = os.path.join(args.working_dir, 'params_nbinom.csv')

    # id,size,prob,interarrival_mean,interarrival_var,interarrival_count
    df_nbinom_params = pd.read_csv(nbinom_filename)

    # We could check args.draw_plots to see if that's active, but
    # presumably anyone who calls this function wants to see the plots...

    def r2(x, y):
        return stats.pearsonr(x, y)[0] ** 2

    r2_value = r2(df_nbinom_params['r'].values, df_nbinom_params['prob'].values)

    if args.draw_plots:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='r', y='prob',
                        size='interarrival_count', data=df_nbinom_params)
        plt.title('Negative binomial parameter space (r vs p), $R^2$=%.2f' % r2_value)

        plt.figure(figsize=(8, 6))
        x = 1.0 / df_nbinom_params['r'].values
        y = (1.0 - df_nbinom_params['prob'].values) / (df_nbinom_params['prob'].values)
        r2_value = r2(x, y)
        sns.scatterplot(x, y, size=df_nbinom_params['interarrival_count'])
        plt.title('Negative binomial parameter space (r vs p), $R^2$=%.2f, foo' % r2_value)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='interarrival_mean', y='prob',
                        size='interarrival_count', data=df_nbinom_params)
        plt.title('Negative binomial parameter space (mean vs prob)')

        plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
