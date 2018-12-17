import utils_sickle_stats as utils
import numpy as np
import pandas as pd
import os
import datetime
import pytz
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
logger = utils.logger


def main(args):
    logger.info('==================================')
    logger.info('PLOT FIGURES')

    pix_dir = os.path.join(args.working_dir, 'figures')
    utils.safe_mkdir(pix_dir)

    filename = os.path.join(
        args.confidential_dir, args.clean_file)
    df = pd.read_csv(filename)

    if True:
        # Plot length of time of after transfusion until SPRT confidence is
        # achieved.
        df_treated = df.iloc[df.patient_treated.values]

        transfusion_dates = np.sort(
            np.unique(df_treated.infusion_epoch.values))
        transfusion_dates = [datetime.datetime.fromtimestamp(
            t, tz=pytz.utc) for t in transfusion_dates]
        end_date = datetime.datetime(2018, 10, 1)

        transfusion_dates_mpl = [mdates.date2num(t) for t in transfusion_dates]
        end_date_mpl = 6 * [mdates.date2num(end_date)]

        date_duration = [end_date_mpl[i] - transfusion_dates_mpl[i]
                         for i in range(6)]
        plt.figure(figsize=(8, 6))
        plt.barh(1 + np.arange(len(transfusion_dates)), date_duration,
                 left=transfusion_dates_mpl,
                 alpha=.7
                 )
        d1 = mdates.date2num(datetime.datetime(2018, 1, 31))
        plt.plot([d1, d1], [0, 6.5], ':', label='95% confidence')

        d2 = mdates.date2num(datetime.datetime(2018, 2, 26))
        plt.plot([d2, d2], [0, 6.5], ':', label='99% confidence')
        plt.legend()
        plt.ylabel('Subject')
        plt.xlabel('Time since transfusion')
        plt.ylim([.5, 6.5])
        # We need to tell matplotlib that these are dates...
        ax = plt.gca()
        ax.xaxis_date()
        # Rotate date ticks so they're legible
        plt.gcf().autofmt_xdate()

        ax.axis('tight')

        filename = os.path.join(pix_dir, 'SPRT_vanilla.png')
        plt.savefig(filename)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
