import utils_sickle_stats as utils
import numpy as np
import pandas as pd
import os
import datetime
import pytz
from scipy.special import binom
from scipy.stats import beta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
logger = utils.logger


def main(args):
    logger.info('==================================')
    logger.info('PLOT ORDER STATISTICS FIGURES')

    pix_dir = os.path.join(args.working_dir, 'figures')
    utils.safe_mkdir(pix_dir)

    population_sizes = np.arange(2, 21)
    y = np.zeros(len(population_sizes))
    plt.figure(figsize=(10, 8))
    Q25_list = [.6, .7, .8, .9, 1.0]
    for Q25 in Q25_list:
        y = np.zeros(len(population_sizes))
        for i, pop_size in enumerate(population_sizes):
            for j in range(pop_size + 1):
                # p1 = prob that j people have R<0.25:
                p1 = (1.0 * binom(pop_size, j) *
                      np.power(Q25, j) * np.power(1.0 - Q25, pop_size - j))
                # probability that j-th entry is >=median
                credibility = beta.cdf(0.50, j + 1, pop_size - j + 1)
                if credibility < 0.05 and np.isfinite(p1):
                    y[i] += p1
        plt.plot(population_sizes, y, '-o', label="$Q_{25}$=%.2f" % (Q25))
    plt.xlabel('Population size')
    plt.ylabel('Probability of acceptance')
    plt.xlim([np.min(population_sizes), np.max(population_sizes)])
    plt.title('Median VOE duration reduction >75% (with 95% credibility)')
    plt.legend(loc=4)

    filename = os.path.join(pix_dir, 'order_stats.png')
    # plt.show()
    plt.savefig(filename)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
