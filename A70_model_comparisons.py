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
    nbinom_filename = os.path.join(args.working_dir, 'params_nbinom_count.csv')
    df_nbinom_params = pd.read_csv(nbinom_filename)
    nbinom_llf = df_nbinom_params.loglikelihood.values

    tweedie_filename = os.path.join(
        args.working_dir, 'params_tweedie_count.csv')
    df_tweedie_params = pd.read_csv(tweedie_filename)
    tweedie_llf = df_tweedie_params.loglikelihood.values

    poisson_filename = os.path.join(
        args.working_dir, 'params_poisson_count.csv')
    df_poisson_params = pd.read_csv(poisson_filename)
    poisson_llf = df_poisson_params.loglikelihood.values

    if args.draw_plots:
        plt.figure()

        # Neg binom vs Tweedie
        sns.scatterplot(nbinom_llf, tweedie_llf, label='Tweedie')
        sns.scatterplot(nbinom_llf, poisson_llf, label='Poisson')

        MM = max(np.max(nbinom_llf), np.max(tweedie_llf), np.max(poisson_llf))
        plt.plot([0, MM], [0, MM], ':')
        plt.title('Log-Likelihoods')
        plt.xlabel('Negative Loglikelihood (Neg Binom)')
        plt.ylabel('Negative Loglikelihood (Competing Model)')
        plt.title('Log-Likelihoods')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
