import h5py
import scipy.stats as st
from collections import defaultdict
import numpy as np
import pandas as pd
import copy
import lasagne


filename_whas = "data/whas/whas_train_test.h5"
filename_metabric = "data/metabric/metabric_IHC4_clinical_train_test.h5"
filename = filename_whas

datasets = defaultdict(dict)
with h5py.File(filename, 'r') as fp:
    for ds in fp:
        for array in fp[ds]:
            datasets[ds][array] = fp[ds][array][:]

dataset = datasets['train']

covariates = pd.DataFrame(dataset['x'])
time = pd.DataFrame(dataset['t'], columns=['time'])
status = pd.DataFrame(dataset['e'], columns=['status'])
df = pd.concat([time, status, covariates], axis=1)
