import numpy as np
import pandas as pd
import os
import utils_sickle_stats as utils
import scipy.stats as stats
import sys
if sys.version_info.major == 2:
    import cPickle as pickle
else:
    import pickle

logger = utils.logger


def main(args):
    logger.info('==================================')
    logger.info('SYNTHETIC DATA')

    outfilename_sparse = os.path.join(
        args.working_dir, 'synthetic_nbd_sparse.csv')
    outfilename_dense = os.path.join(
        args.working_dir, 'synthetic_nbd_dense.csv')

    if os.path.exists(outfilename_dense) and os.path.exists(outfilename_sparse):
        logger.info('Output file already exists; skipping computation')
        return

    filename = os.path.join(args.working_dir, 'priors_nbinom.pkl')
    prior_models = pickle.load(open(filename, 'rb'))

    # How many (synthetic) subjects?
    N = 1000

    # How much data do we generate per subject?
    days_before_transfusion = 2 * 365
    days_after_transfusion = 6 * 365

    window_days = 7
    # Sparse data frame (just track when events happen)
    # df_syn_sparse = pd.DataFrame(
    #     columns=['subject', 'measurement_period', 'pretransfusion', 'duration_days', 'true_r', 'true_prob'])
    # Dense data frame (status for each day; about 400K rows)
    rows_before = int(days_before_transfusion / window_days)
    rows_after = int(days_after_transfusion / window_days)
    df_syn_dense = pd.DataFrame(columns=['subject', 'week', 'pretransfusion', 'events', 'days_in_hospital',
                                         'cumu_events', 'cumu_days_in_hospital', 'true_p', 'true_r'],
                                index=range(N * (rows_before + rows_after)))

    (beta_alpha, beta_beta, gamma_alpha, gamma_scale) = prior_models['vanilla']

    dense_row_index = 0
    for subj in range(N):
        subj_string = "S%06d" % subj
        p_before = stats.beta.rvs(beta_alpha, beta_beta)
        r_before = stats.gamma.rvs(gamma_alpha, scale=gamma_scale)

        # Use "randomly distributed colonies" interpretation to generate hospital days
        # per Kozubowski and Podgorski, 2009
        for pretransfusion, row_size, p, r in [(True, rows_before, p_before, r_before),
                                               (False, rows_after, p_before, 0.25 * r_before)]:

            lambda_param = - r * np.log(p)
            max_duration = 2048
            duration_prob = np.zeros(max_duration)
            for k in range(1, max_duration):
                duration_prob[k] = -np.power((1.0 - p), k) / (np.log(p) * k)
            duration_cumu_prob = np.cumsum(duration_prob)
            # assert (duration_cumu_prob[-1]) > .99
            duration_cumu_prob[-1] = 1.0

            VOE_event_counts = stats.poisson.rvs(
                lambda_param, size=row_size)
            VOE_days_in_hospital_counts = np.zeros(row_size)
            MM = np.max(VOE_event_counts)
            for k in range(1, MM + 1):
                indices = (VOE_event_counts == k)
                VOE_days_in_hospital_counts[indices] += np.searchsorted(
                    duration_cumu_prob, np.random.rand(np.sum(indices)))

            # If more than 7 hospitalization days in a week, need to smooth it
            # out a little.
            while (True):
                bad_rows = VOE_days_in_hospital_counts[:-1] > 7
                if np.sum(bad_rows) == 0:
                    break
                excess = VOE_days_in_hospital_counts[:-1][bad_rows] - 7
                VOE_days_in_hospital_counts[:-1][bad_rows] = 7
                VOE_days_in_hospital_counts[1:][bad_rows] += excess
                # logger.info('Excess: %s %s %d' %
                #            (subj, pretransfusion, np.sum(bad_rows)))
            if VOE_days_in_hospital_counts[-1] > 7:
                logger.info('Terminal excess: %s %s %d' % (
                    subj, pretransfusion, VOE_days_in_hospital_counts[-1]))
                VOE_days_in_hospital_counts[-1] = 7

            df_syn_dense['subject'].values[
                dense_row_index:dense_row_index + row_size] = subj_string
            df_syn_dense['week'].values[
                dense_row_index:dense_row_index + row_size] = np.arange(row_size)
            df_syn_dense['pretransfusion'].values[
                dense_row_index:dense_row_index + row_size] = pretransfusion
            df_syn_dense['events'].values[
                dense_row_index:dense_row_index + row_size] = VOE_event_counts
            df_syn_dense['days_in_hospital'].values[
                dense_row_index:dense_row_index + row_size] = VOE_days_in_hospital_counts.astype(int)
            df_syn_dense['cumu_events'].values[
                dense_row_index:dense_row_index + row_size] = np.cumsum(VOE_event_counts)
            df_syn_dense['cumu_days_in_hospital'].values[
                dense_row_index:dense_row_index + row_size] = np.cumsum(VOE_days_in_hospital_counts).astype(int)
            df_syn_dense['true_p'].values[
                dense_row_index:dense_row_index + row_size] = p
            df_syn_dense['true_r'].values[
                dense_row_index:dense_row_index + row_size] = r

            dense_row_index += row_size

    # df_syn_sparse.to_csv(outfilename_sparse, index=False)
    df_syn_dense.to_csv(outfilename_dense, index=False)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
