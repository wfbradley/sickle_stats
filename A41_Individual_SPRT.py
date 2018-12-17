# Start with daily_data produced by A03_convert_to_daily_data.py

# Same analysis as in A40*, but now performed by individual.

import math
import numpy as np
import pandas as pd
from pandas import DataFrame
import os
import utils_sickle_stats as utils
logger = utils.logger

pd.set_option('max_colwidth', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1200)

def main(args):
    logger.info('==================================')
    logger.info('SPRT CONDUCTED PER INDIVIDUAL')

    daily_filename = os.path.join(args.confidential_dir, args.daily_file)
    daily_df = pd.read_csv(daily_filename,
                           index_col=0)
    
    patient_ids = sorted(list(set([c[:4] for c in daily_df.columns]))) # sorted for repeatability
    
    combined_df = DataFrame(index=daily_df.index) # DataFrame where we'll put combined results
    
    # We have four categories of events that can happen each day: a patient can be
    # 1.) In hospital before treatment
    # 2.) Out of hospital before treatment
    # 3.) In hospital after treatment
    # 4.) Out of hospital after treatment
    
    for patient in patient_ids:
        OOH_Before_col = [c for c in daily_df.columns if c.endswith('OOH_Before_Cum') & c.startswith(patient)][0]
        OOH_After_col = [c for c in daily_df.columns if c.endswith('OOH_After_Cum') & c.startswith(patient)][0]
        IH_Before_col = [c for c in daily_df.columns if c.endswith('IH_Before_Cum') & c.startswith(patient)][0]
        IH_After_col = [c for c in daily_df.columns if c.endswith('IH_After_Cum') & c.startswith(patient)][0]
        patient_df = daily_df[[OOH_Before_col, OOH_After_col, IH_Before_col, IH_After_col]].copy(deep=True)
        patient_df.columns = [c[5:] for c in patient_df.columns]
       
        # Only need the part if the data in which the patient appeared, so delete rows with all zeros:
        patient_df['In_Study_Yet'] = (patient_df > 0).any(axis = 1)
        patient_df = patient_df[patient_df.In_Study_Yet]
    
        # Need an estimate of this patient's Bernoulli probability of being in hospital before treatment, which is just the observed days in hospital / total days
        days_in_hospital_before_treatment = patient_df.IH_Before_Cum.max()
        days_out_of_hospital_before_treatment = patient_df.OOH_Before_Cum.max()
        p_before_est = days_in_hospital_before_treatment*1.0/(days_in_hospital_before_treatment + days_out_of_hospital_before_treatment)
        assert(p_before_est > 0)
        
        # Now let's assume that the treatment reduces the Bernoulli probability to 25% of the before treatment probability of being in hospital
        p_after_est = 0.25*p_before_est

        # And, finally, the SPRT.  Note this is spelled out pretty much exactly in the book Sequential Analysis by A. Wald on p. 39.
        # In this case our null hypothesis H0 is that, after treatment, the probability of being in hospital is still p_before_est.  
        # The alternative hypothesis H1 is that the probability is p_after_est.
    
        patient_df["cum_sum_log_likelihood"] = math.log(p_after_est/p_before_est)*patient_df["IH_After_Cum"] + \
            math.log((1-p_after_est)/(1-p_before_est))*patient_df["OOH_After_Cum"]
        
        # Insert NaNs before we saw any "after treatment data"
        patient_df.loc[patient_df["IH_After_Cum"]+patient_df["OOH_After_Cum"]==0,"cum_sum_log_likelihood"]=np.nan

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
        patient_df["SPRT_.05"]=patient_df["cum_sum_log_likelihood"].apply(SPRT)
        
    
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
        patient_df["SPRT_.01"]=patient_df["cum_sum_log_likelihood"].apply(SPRT)
        
        patient_df.drop(columns="In_Study_Yet", inplace=True)
        patient_df.columns = [patient+"_"+c for c in patient_df.columns]
        
        combined_df = pd.concat([combined_df,patient_df], axis=1, sort=False)
        
    combined_filename = os.path.join(args.confidential_dir, "individual_SPRT_results.csv")
    combined_df.to_csv(combined_filename)
        
if __name__ == '__main__':
    args = utils.parse_arguments()
    utils.initialize_logger(args)
    main(args)
