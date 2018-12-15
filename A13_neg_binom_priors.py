import pandas as pd
import os
import utils_sickle_stats as utils
import cPickle as pickle
import scipy.stats as stats

logger = utils.logger


def main(args):
    logger.info('==================================')
    logger.info('ESTIMATE NEG BINOM PRIORS')

    filename = os.path.join(args.working_dir, 'params_nbinom_count.csv')
    # id,r,prob,count_mean,count_var,count_len,loglikelihood
    df_neg_binom = pd.read_csv(filename)

    models = {}

    # Simple model for priors on parameters:
    #   r as Gamma
    #   p as beta

    (beta_alpha, beta_beta, _, _) = stats.beta.fit(
        df_neg_binom.prob.values, floc=0, fscale=1)
    (gamma_alpha, gamma_loc, gamma_scale) = stats.gamma.fit(
        df_neg_binom.r.values, floc=0)

    # independently as Gaussians
    models['vanilla'] = (beta_alpha, beta_beta, gamma_alpha, gamma_scale)

    filename = os.path.join(args.working_dir, 'priors_nbinom.pkl')
    pickle.dump(models, open(filename, 'w'))


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
