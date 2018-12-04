# Original data is stored as an Excel spreadsheet.
# Add new columns to flag data anomalies and add some
# utility columns; store output as CSV file.  Read and
# write to "confidential_data/" directory.

import numpy as np
import pandas as pd
import argparse
import os
import time


# Convert string "2014-11-02" to epoch time
def to_epoch(s):
    t = time.strptime(s, "%Y-%m-%d")
    return(time.mktime(t))


def clean_data(args):

    input_filename = os.path.join(args.confidential_data_dir, args.input_file)

    # [u'Unique Subject Identifier', u'Informed consent date',
    #  u'Infusion Date', u'Type of VOE', u'Onset date', u'Resolution date',
    #  u'No. of VOEs']
    df = pd.read_excel(input_filename)

    # We will leave the original columns unchanged, and only add derived
    # columns

    # Distinguish data from earlier, less-effective therapy.  Will be false if
    # therapy occured too long ago, or if no therapy was applied.
    df['patient_treated'] = False
    for i in xrange(len(df)):
        df['patient_treated'].values[i] = (
            (int(df['Unique Subject Identifier'].values[i][-4:]) >= 1315) and
            (pd.notnull(df['Infusion Date'].values[i])))

    # Standardize dates for episode onset/resolution into epoch seconds, and record
    # duration of event in days.  Note that if, e.g., onset date = resolution date, then
    # this episode lasts 1 day, not zero days.
    #
    # If a field is missing, value is set to NaN.

    df['start_epoch'] = 0.0
    df['end_epoch'] = 0.0
    df['episode_duration_days'] = 0.0

    for i in xrange(len(df)):
        start_date = df['Onset date'].values[i]
        end_date = df['Resolution date'].values[i]

        # A well-formatted date looks like '2018-01-15';
        # partial dates (like just '2018-01' or '2018') exist.
        for col, d in [('start_epoch', start_date), ('end_epoch', end_date)]:
            if len(d.split('-')) == 3:
                df[col].values[i] = to_epoch(d)
            else:
                df[col].values[i] = np.NaN
        df['episode_duration_days'].values[i] = 1 + (
            df['end_epoch'].values[i] - df['start_epoch'].values[i]) / 86400

    output_filename = os.path.join(
        args.confidential_data_dir, args.output_file)
    print "Writing file ", output_filename
    df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised ML classification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--confidential_data_dir', type=str, default='confidential_data',
                        help='Directory for reading and writing confidential data.')
    parser.add_argument('--input_file', type=str, default='l_voc_acs_mh.xlsx',
                        help='Input Excel data file within confidential_data_dir')
    parser.add_argument('--output_file', type=str, default='cleaned_data.csv',
                        help='Input Excel data file within confidential_data_dir')
    args = parser.parse_args()

    clean_data(args)
