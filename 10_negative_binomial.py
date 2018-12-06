import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import fit_nbinom
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

logger = utils.logger
sns.set()


def negative_binomial_analysis(args):

    logger.info('==================================')
    logger.info('NEGATIVE BINOMIAL')

    interarrival_filename = os.path.join(
        args.confidential_dir, "interarrival_times.csv")
    # id,interarrival_days
    df_interarrival = pd.read_csv(interarrival_filename)
    grouped_interarrival = df_interarrival.groupby('id')

    df_nbinom_params = pd.DataFrame(columns=['id', 'size', 'prob', 'interarrival_mean',
                                             'interarrival_var', 'interarrival_count'],
                                    index=np.arange(len(grouped_interarrival)))

    for i, (subject, df_subject) in enumerate(grouped_interarrival):
        # nbinom_params contains "size" and "prob"
        data = df_subject['interarrival_days'].values
        df_nbinom_params['id'].values[i] = subject
        df_nbinom_params['interarrival_mean'].values[i] = np.mean(data)
        df_nbinom_params['interarrival_var'].values[i] = np.var(data)
        df_nbinom_params['interarrival_count'].values[i] = len(data)

        try:
            nbinom_params = fit_nbinom.fit_nbinom(data)
        except Exception:
            logger.info('Could not fit negative binomial on %s' % subject)

        (n, p) = (nbinom_params['size'], nbinom_params['prob'])

        df_nbinom_params['id'].values[i] = subject
        df_nbinom_params['size'].values[i] = n
        df_nbinom_params['prob'].values[i] = p

        if args.draw_plots:
            plt.figure(figsize=(8, 6))
            interarrival_times = df_subject['interarrival_days'].values
            sns.distplot(interarrival_times, rug=True)
            plt.xlim(0, np.max(data) + 20)

            time_span_days = np.sum(data)
            plt.title('Subject %s; N=%d episodes over %d days (%d days/eps)' % (
                subject[-4:], len(df_subject) + 1,
                time_span_days, 1.0 * time_span_days / len(df_subject)))
            plt.xlabel('Episode interarrival time in days')

            domain = np.arange(1, np.max(data) + 20)
            rv = scipy.stats.nbinom(n, p)
            nbinom_density = rv.pmf(domain)
            plt.plot(domain, nbinom_density,
                     label='Negative Binomial (n=%.2f, p=%.2f)' % (n, p))

            plt.legend()
            plt.show()

    print "sfefessfe"
    df_nbinom_params.to_csv(os.path.join(
        args.working_dir, 'params_nbinom.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    negative_binomial_analysis(args)
