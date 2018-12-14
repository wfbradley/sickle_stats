# Start with "cleaned_data" produced by A00_clean_data.py
# to avoid re-doing much of Bill's work (less well).

import pandas as pd
from pandas import DataFrame
import os
import utils_sickle_stats as utils
logger = utils.logger
pd.set_option('max_colwidth', 100)


def main(args):
    logger.info('==================================')
    logger.info('CONVERSION TO DAILY SEQUENCE OF DATA FOR SPRT')

    cleaned_filename = os.path.join(args.confidential_dir, args.clean_file)

    # ['Unique Subject Identifier', 'Informed consent date', 'Infusion Date',
    #   'Type of VOE', 'Onset date', 'Resolution date', 'No. of VOEs',
    #   'patient_treated', 'start_epoch', 'end_epoch', 'episode_duration_days',
    #   'start_epoch_coalesced', 'end_epoch_coalesced',
    #   'episode_duration_days_coalesced']
    clean_df = pd.read_csv(cleaned_filename)

    # The same event may be categorized as ACS and VOC, leading to duplicate
    # rows; uniquify VOEs
    clean_df = utils.filter_cleaned_data(clean_df)

    # To get going quickly, let's take two shortcuts, to possibly clean up later:
    # 1.) Drop rows with partial dates
    # 2.) Drop untreated patients - not sure if this is a good idea, but I
    # think those patients' data are probably right-censored
    valid_df = clean_df[(clean_df.start_epoch.notnull()) & (clean_df.end_epoch.notnull()) & (
        clean_df["Type of VOE"] == 'VOC') & (clean_df.patient_treated)]

    # Make sure data is sorted by patient and then start_epoch
    valid_df = valid_df.sort_values(
        by=['Unique Subject Identifier', 'start_epoch'])

    min_date = pd.to_datetime(min(valid_df["Infusion Date"].min(), valid_df[
                              "Onset date"].min()))  # Estimated earliest date we need
    # Today happens to be 12/11/18
    rng = pd.date_range(start=min_date, end='2018-12-11')
    # Create an empty daily dataframe
    daily_df = DataFrame(index=rng)
    for i in pd.unique(valid_df["Unique Subject Identifier"]):
        suffix = i[-4:]
        daily_df[suffix + "_OOH_Before"] = 0
        daily_df[suffix + "_IH_Before"] = 0
        daily_df[suffix + "_OOH_After"] = 0
        daily_df[suffix + "_IH_After"] = 0

    # Loop over the patients and set values.
    for i in pd.unique(valid_df["Unique Subject Identifier"]):
        suffix = i[-4:]
        temp_df = valid_df[valid_df["Unique Subject Identifier"] == i]
        first = True
        for _, row in temp_df.iterrows():
            if first:
                first = False  # N.B. We skip the first episode to conservatively bias the proportion of time
                # in hospital before treatment down (and to compensate for the
                # left-censoring of the data).
                earliest_resolution_date = row['Resolution date']
                continue
            start = row['Onset date']
            end = row['Resolution date']
            infusion = row['Infusion Date']
            before = infusion > start
            rng_temp = pd.date_range(start=start, end=end)
            if before:
                for d in rng_temp:
                    daily_df.loc[d][suffix + "_IH_Before"] = 1
            else:  # We don't see this case in our data, but for completeness' sake.
                for d in rng_temp:
                    daily_df.loc[d][suffix + "_IH_After"] = 1
        # Now set values for when patients were out of hospital.
        # To deal with the fact that I want to iterate over daily_df while
        # changing it
        temp_copy = daily_df.copy(deep=True)
        for day, row in temp_copy.iterrows():
            if (str(day)[:10]) <= earliest_resolution_date:
                continue
            before = (str(day)[:10] < infusion)
            if before:
                daily_df.loc[day][suffix + "_OOH_Before"] = 1 - \
                    daily_df.loc[day][suffix + "_IH_Before"]
            else:  # We don't see this case in our data, but for completeness' sake.
                daily_df.loc[day][suffix + "_OOH_After"] = 1 - \
                    daily_df.loc[day][suffix + "_IH_After"]

    # We only estimated the start date before.  This next bit lops off the top
    # of the dataframe that we don't need.
    daily_df["row_sum"] = daily_df.sum(axis=1)
    # We only estimated the start date before.  This lops off the top of the
    # dataframe that we don't need.
    daily_df = daily_df[daily_df.row_sum > 0]
    daily_df.drop(labels="row_sum", axis=1, inplace=True)

    # To calculate proportions for SPRT later, it will be nice to have
    # cumulative sums before and after the treatment per patient.
    for i in pd.unique(valid_df["Unique Subject Identifier"]):
        suffix = i[-4:]
        for cat in ["_OOH_Before", "_IH_Before", "_OOH_After", "_IH_After"]:
            daily_df[suffix + cat + "_Cum"] = daily_df[suffix + cat].cumsum()

    daily_filename = os.path.join(args.confidential_dir, "daily_file.csv")
    daily_df.to_csv(daily_filename)


if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
