import pandas as pd
import os
import utils_sickle_stats as utils
import matplotlib.pyplot as plt
import seaborn as sns

logger = utils.logger
sns.set()


def negative_binomial_parameter_visualization(args):

    logger.info('==================================')
    logger.info('NEGATIVE BINOMIAL PARAMETER SPACE VISUALIZATION')

    nbinom_filename = os.path.join(args.working_dir, 'params_nbinom.csv')

    # id,size,prob,interarrival_mean,interarrival_var,interarrival_count
    df_nbinom_params = pd.read_csv(nbinom_filename)

    # We could check args.draw_plots to see if that's active, but
    # presumably anyone who calls this function wants to see the plots...

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='size', y='prob',
                    size='interarrival_count', data=df_nbinom_params)
    plt.title('Negative binomial parameter space (size v prob)')

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='interarrival_mean', y='prob',
                    size='interarrival_count', data=df_nbinom_params)
    plt.title('Negative binomial parameter space (mean v prob)')

    plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    negative_binomial_parameter_visualization(args)
