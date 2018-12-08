# Statistical Modeling of Sickle Cell Disease

##### Table of Contents  
-  [The Problem](#the-problem) 
-  [Installation](#installation)
-  [Data](#data) 
-  [Models](#models) 
    * [Hierarchical Poisson](#hierarchical-poisson)
    * [Hierarchical Negative Binomial Process](#hierarchical-negative-binomial-process) 
-  [Generative Models](#generative-models)
-  [Author](#author)  
-  [License](#license)  
-  [Acknowledgements](#acknowledgements)  


## The Problem

We wish to determine if an experimental treatment for sickle cell disease (SCD)
is effective.  Patients with SCD suffer from vaso-occlusive episodes (VOEs),
which can be severe enough to require hospitalization.  The treatment occurs at
a single point in time, and we wish to determine if the rate of episodes 
after the treatment is significantly lower than the rate before the treatment.

(Because the treatment is invasive, ethical considerations prevent a double-blinded experiment.)

## Installation

We assume you have already installed Python3 and PIP3.

To download this repo and the required libraries:
```
git clone https://github.com/wfbradley/sickle_stats.git
cd sickle_stats
pip3 install -r requirements.txt --user --upgrade
```

If the `pip install` fails for permission reasons, one can instead try
```
sudo -H pip3 install -r requirements.txt --upgrade
```

## Data
Data is confidential so cannot be provided in GitHub.  All confidential
data should be put in the "confidential_data/" subdirectory.

The data is structured as
follows...

Data is cleaned and put into a standard form by running
```
python 00_clean_data.py
```

## Models

The rate of episodes differs between patients, so many models have a
hierarchical structure.  So, given a patient, we first sample the severity of
the disease for that individual; then conditional on the severity, we
sample the time of the episodes for that patient.

### Hierarchical Poisson

Episode process for patient i modeled as Poisson process of rate
![lambda_i](http://mathurl.com/yd2xhu3q.png).  Distribution of
![lambda_i](http://mathurl.com/yd2xhu3q.png) is, say, a Gamma(r,alpha)
distribution.

### Hierarchical Negative Binomial Process

Episode process for patient i modeled as a [negative binomial
process](https://en.wikipedia.org/wiki/Negative_binomial_distribution) with
parameters ![NB(r_i,p_i)](http://mathurl.com/yca7w4ce.png).  The r_i can be
sampled from another negative binomial, and p_i can be sampled from a beta
distribution. (These should probably be correlated.)

## Generative Models

Once we have fit a probabilistic model, we can generate samples from it to
produce a set of synthetic data for further analysis...

## Authors

This code was conceived and written by William Bradley and Karl Knaub in 2018.

## License

This project is licensed under the MIT License.  See `LICENSE` file for the
complete license.

Copyright (c) 2018 William Bradley and Karl Knaub

## Acknowledgements



