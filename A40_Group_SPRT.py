# Start with daily_data produced by A03_convert_to_daily_data.py

import math
import numpy as np
import pandas as pd
#from pandas import DataFrame
import os
import utils_sickle_stats as utils
logger = utils.logger

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1200)

def main(args):
    logger.info('==================================')
    logger.info('SPRT CONDUCTED ON WHOLE POPULATION')

    daily_filename = os.path.join(args.confidential_dir, args.daily_file)
    daily_df = pd.read_csv(daily_filename,
                           index_col=0)
    
    # We have four categories of events that can happen each day: a patient can be
    # 1.) In hospital before treatment
    # 2.) Out of hospital before treatment
    # 3.) In hospital after treatment
    # 4.) Out of hospital after treatment
    # Since our first stab at using SPRT is going to look at the whole population,
    # let's sum the data across each of those categories
    
    OOH_Before_cols = [c for c in daily_df.columns if c.endswith('OOH_Before')]
    OOH_After_cols = [c for c in daily_df.columns if c.endswith('OOH_After')]
    IH_Before_cols = [c for c in daily_df.columns if c.endswith('IH_Before')]
    IH_After_cols = [c for c in daily_df.columns if c.endswith('IH_After')]
    agg_df = daily_df.copy(deep=True)
    agg_df["OOH_Before_Sum"] = daily_df[OOH_Before_cols].sum(axis=1)
    agg_df["OOH_After_Sum"] = daily_df[OOH_After_cols].sum(axis=1)
    agg_df["IH_Before_Sum"] = daily_df[IH_Before_cols].sum(axis=1)
    agg_df["IH_After_Sum"] = daily_df[IH_After_cols].sum(axis=1)
    agg_df = agg_df[["OOH_Before_Sum", "OOH_After_Sum", "IH_Before_Sum", "IH_After_Sum"]]
    
    # Now cumulatively sum these to make later calculations easier
    agg_df["OOH_Before_Sum_Cum"] = agg_df["OOH_Before_Sum"].cumsum()
    agg_df["OOH_After_Sum_Cum"] = agg_df["OOH_After_Sum"].cumsum()
    agg_df["IH_Before_Sum_Cum"] = agg_df["IH_Before_Sum"].cumsum()
    agg_df["IH_After_Sum_Cum"] = agg_df["IH_After_Sum"].cumsum()
    
    # Need a running best estimate of the Bernoulli probability of being in hospital before treatment, which is just the observed days in hospital / total days
    agg_df["p_before_est"] = agg_df["IH_Before_Sum_Cum"]*1.0/(agg_df["IH_Before_Sum_Cum"] + agg_df["OOH_Before_Sum_Cum"])
    
    # Now let's assume that the treatment reduces the Bernoulli probability to 25% of the before treatment probability of being in hospital
    agg_df["p_after_est"] = 0.25*agg_df["p_before_est"]
    
    # And, finally, the SPRT.  Note this is spelled out pretty much exactly in the book Sequential Analysis by A. Wald on p. 39.
    # In this case our null hypothesis H0 is that, after treatment, the probability of being in hospital is still p_before_est.  
    # The alternative hypothesis H1 is that the probability is p_after_est.
    
    agg_df["cum_sum_log_likelihood"] = agg_df["IH_After_Sum_Cum"]*((agg_df["p_after_est"]/agg_df["p_before_est"]).apply(math.log)) + \
        agg_df["OOH_After_Sum_Cum"]*((1-agg_df["p_after_est"])/(1-agg_df["p_before_est"])).apply(math.log)
    # Insert NaNs before we saw any "after treatment data"
    agg_df[agg_df["IH_After_Sum_Cum"]+agg_df["OOH_After_Sum_Cum"]==0]["cum_sum_log_likelihood"]=np.nan

    # 5% significance level / 95% power, i.e. run SPRT with a 5% chance of making a Type I error and a 5% chance of making a Type II error
    alpha = .05 # Type I error rate
    beta = .05 # Type II error rate
    a = math.log(beta/(1-alpha))
    b = math.log((1-beta)/alpha)
    def SPRT(x):
        if x >= b:
            return "H1"
        if x <= a:
            return "H0"
        if pd.isna(x):
            return np.nan
        else:
            return "Inconclusive"
    agg_df["SPRT_.05"]=agg_df["cum_sum_log_likelihood"].apply(SPRT)
    
    # 1% significance level / 99% power, i.e. run SPRT with a 1% chance of making a Type I error and a 5% chance of making a Type II error
    alpha = .01 # Type I error rate
    beta = .01 # Type II error rate
    a = math.log(beta/(1-alpha))
    b = math.log((1-beta)/alpha)
    def SPRT(x):
        if x >= b:
            return "H1"
        if x <= a:
            return "H0"
        if pd.isna(x):
            return np.nan
        else:
            return "Inconclusive"
    agg_df["SPRT_.01"]=agg_df["cum_sum_log_likelihood"].apply(SPRT)
    
    print(agg_df[["SPRT_.05","SPRT_.01"]].head(1200))
    
if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
