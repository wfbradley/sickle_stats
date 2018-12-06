import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import tweedie_dist
import tweedie_fit
import matplotlib.pyplot as plt
import seaborn as sns

logger = utils.logger
sns.set()


def tweedie_analysis(args):

    logger.info('==================================')
    logger.info('TWEEDIE FIT')

    interarrival_filename = os.path.join(
        args.confidential_dir, "interarrival_times.csv")
    # id,interarrival_days
    df_interarrival = pd.read_csv(interarrival_filename)
    grouped_interarrival = df_interarrival.groupby('id')

    df_tweedie_params = pd.DataFrame(columns=['id', 'mu', 'p', 'phi', 'interarrival_mean',
                                              'interarrival_var', 'interarrival_count', 'loglikelihood'],
                                     index=np.arange(len(grouped_interarrival)))

    for i, (subject, df_subject) in enumerate(grouped_interarrival):
        logger.info("Subject %d of %d" % (i+1, len(grouped_interarrival)))
        data = df_subject['interarrival_days'].values

        df_tweedie_params['id'].values[i] = subject
        df_tweedie_params['interarrival_mean'].values[i] = np.mean(data)
        df_tweedie_params['interarrival_var'].values[i] = np.var(data)
        df_tweedie_params['interarrival_count'].values[i] = len(data)

        try:
            tweedie_params = tweedie_fit.tweedie_fit(data)
            (mu, p, phi, loglikelihood) = (tweedie_params['mu'], tweedie_params['p'],
                                           tweedie_params['phi'], tweedie_params['loglikelihood'])
        except Exception:
            logger.info('Could not fit Tweedie on %s' % subject)
            (mu, p, phi, loglikelihood) = (np.nan, np.nan, np.nan, np.nan)

        df_tweedie_params['mu'].values[i] = mu
        df_tweedie_params['p'].values[i] = p
        df_tweedie_params['phi'].values[i] = phi
        df_tweedie_params['loglikelihood'].values[i] = loglikelihood

        if np.isnan(mu):
            continue

        if args.draw_plots:
            plt.figure(figsize=(8, 6))
            interarrival_times = df_subject['interarrival_days'].values
            sns.distplot(interarrival_times, rug=True)
            plt.xlim(0, np.max(data) + 20)

            time_span_days = np.sum(data)
            plt.title('Subject %s; N=%d episodes over %d days ($\mu$=%d, $\sigma$=%.1f, $\sigma^2$=%d)' % (
                subject[-4:], len(df_subject) + 1,
                time_span_days, np.mean(interarrival_times),
                np.std(interarrival_times),
                np.var(interarrival_times)))
            plt.xlabel('Episode interarrival time in days')

            domain = np.arange(1, np.max(data) + 20)
            rv = tweedie_dist.tweedie(mu=mu, p=p, phi=phi)
            tweedie_density = rv.pdf(domain)
            plt.plot(domain, tweedie_density,
                     label='Tweedie (mu=%.2f, p=%.2f, phi=%.2f)' % (mu, p, phi))

            plt.legend()
            plt.show()

    df_tweedie_params.to_csv(os.path.join(
        args.working_dir, 'params_tweedie.csv'), index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    tweedie_analysis(args)
