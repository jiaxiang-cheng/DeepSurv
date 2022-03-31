# import sys
# sys.path.append('/deepsurv')
import lasagne
import matplotlib.pyplot as plt
# %matplotlib inline

import deepsurv
import visualize
from deepsurv_logger import TensorboardLogger

import h5py
import scipy.stats as st
from collections import defaultdict
import copy
import lasagne

import utils

import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
import h5py
import uuid
import copy
import json

import sys, os
sys.path.append("/DeepSurv/deepsurv")
import deepsurv


# Force matplotlib to not use any Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import visualize
import utils
from deepsurv_logger import TensorboardLogger

import time
localtime   = time.localtime()
TIMESTRING  = time.strftime("%m%d%Y%M", localtime)


def evaluate_model(model, dataset, bootstrap=False):
    def mse(model):
        def deepsurv_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(model.predict_risk(x))
            return ((hr_pred - hr) ** 2).mean()

        return deepsurv_mse

    metrics = {}

    # Calculate c_index
    metrics['c_index'] = model.get_concordance_index(**dataset)
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(model.get_concordance_index, dataset)

    # Calculate MSE
    if 'hr' in dataset:
        metrics['mse'] = mse(model)(**dataset)
        if bootstrap:
            metrics['mse_bootstrap'] = utils.bootstrap_metric(mse(model), dataset)

    return metrics


def dataframe_to_deepsurv_ds(df, event_col='Event', time_col='Time'):
    # Extract the event and time columns as numpy arrays
    e = df[event_col].values.astype(np.int32)
    t = df[time_col].values.astype(np.float32)

    # Extract the patient's covariates as a numpy array
    x_df = df.drop([event_col, time_col], axis=1)
    x = x_df.values.astype(np.float32)

    # Return the DeepSurv dataframe
    return {'x': x, 'e': e, 't': t}


def save_risk_surface_visualizations(model, dataset, norm_vals, output_dir, plot_error, experiment,
                                     trt_idx):
    if experiment == 'linear':
        clim = (-3, 3)
    elif experiment == 'gaussian' or experiment == 'treatment':
        clim = (-1, 1)
    else:
        clim = (0, 1)

    risk_fxn = lambda x: np.squeeze(model.predict_risk(x))
    color_output_file = os.path.join(output_dir, "deep_viz_color_" + TIMESTRING + ".pdf")
    visualize.plot_experiment_scatters(risk_fxn, dataset, norm_vals=norm_vals,
                                       output_file=color_output_file, figsize=(4, 3), clim=clim,
                                       plot_error=plot_error, trt_idx=trt_idx)

    bw_output_file = os.path.join(output_dir, "deep_viz_bw_" + TIMESTRING + ".pdf")
    visualize.plot_experiment_scatters(risk_fxn, dataset, norm_vals=norm_vals,
                                       output_file=bw_output_file, figsize=(4, 3), clim=clim, cmap='gray',
                                       plot_error=plot_error, trt_idx=trt_idx)


def save_treatment_rec_visualizations(model, dataset, output_dir,
                                      trt_i=1, trt_j=0, trt_idx=0):
    trt_values = np.unique(dataset['x'][:, trt_idx])
    print("Recommending treatments:", trt_values)
    rec_trt = model.recommend_treatment(dataset['x'], trt_i, trt_j, trt_idx)
    rec_trt = np.squeeze((rec_trt < 0).astype(np.int32))

    rec_dict = utils.calculate_recs_and_antirecs(rec_trt, true_trt=trt_idx, dataset=dataset)

    output_file = os.path.join(output_dir, '_'.join(['deepsurv', TIMESTRING, 'rec_surv.pdf']))
    print(output_file)
    visualize.plot_survival_curves(experiment_name='DeepSurv', output_file=output_file, **rec_dict)


def save_model(model, output_file):
    model.save_weights(output_file)


# # Read in dataset and print the first five elements to get a sense of what the dataset looks like
# train_dataset_fp = 'example_data.csv'
# train_df = pd.read_csv(train_dataset_fp)
# # train_df.head()
#
# # You can also use this function on your training dataset, validation dataset, and testing dataset
# train_data = dataframe_to_deepsurv_ds(train_df, event_col='Event', time_col='Time')

filename = "./_backup/experiments/data/whas/whas_train_test.h5"

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

# Extract the event and time columns as numpy arrays
e = df['status'].values.astype(np.int32)
t = df['time'].values.astype(np.float32)

# Extract the patient's covariates as a numpy array
x_df = df.drop(['status', 'time'], axis=1)
x = x_df.values.astype(np.float32)

# Return the DeepSurv dataframe
train_data = {'x': x, 'e': e, 't': t}
norm_vals = {
    'mean': datasets['train']['x'].mean(axis=0),
    'std': datasets['train']['x'].std(axis=0)
}

# Set up hyper-parameters
hyperparams = {
    'L2_reg': 10.0,
    'batch_norm': True,
    'dropout': 0.4,
    'hidden_layers_sizes': [25, 25],
    'learning_rate': 1e-05,
    'lr_decay': 0.001,
    'momentum': 0.9,
    'n_in': train_data['x'].shape[1],
    'standardize': True
}

# Create an instance of DeepSurv using the hyper-parameters defined above
model = deepsurv.DeepSurv(**hyperparams)

# DeepSurv can now leverage TensorBoard to monitor training and validation
'''
This section of code is optional. If you don't want to use the tensorboard logger
Uncomment the below line, and comment out the other three lines:
logger = None
'''
experiment_name = 'test_experiment_sebastian'
logdir = './logs/tensorboard/'
logger = TensorboardLogger(experiment_name, logdir=logdir)

# Now we train the model
update_fn = lasagne.updates.nesterov_momentum  # The type of optimizer to use
# Check out http://lasagne.readthedocs.io/en/latest/modules/updates.html for other optimizers to use
n_epochs = 2000

# If you have validation data, you can add it as the second parameter to the function
metrics = model.train(train_data, n_epochs=n_epochs, logger=logger, update_fn=update_fn)

# Print the final metrics
print('Train C-Index:', metrics['c-index'][-1])
# print('Valid C-Index: ',metrics['valid_c-index'][-1])

# Plot the training / validation curves
visualize.plot_log(metrics)
plt.show()


# Evaluate Model
with open(args.model, 'r') as fp:
    json_model = fp.read()
    hyperparams = json.loads(json_model)

train_data = datasets['train']
if hyperparams['standardize']:
    train_data = utils.standardize_dataset(train_data, norm_vals['mean'], norm_vals['std'])

metrics = evaluate_model(model, train_data)
print("Training metrics: " + str(metrics))
if 'valid' in datasets:
    valid_data = datasets['valid']
    if hyperparams['standardize']:
        valid_data = utils.standardize_dataset(valid_data, norm_vals['mean'], norm_vals['std'])
        metrics = evaluate_model(model, valid_data)
    print("Valid metrics: " + str(metrics))

if 'test' in datasets:
    test_dataset = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
    metrics = evaluate_model(model, test_dataset, bootstrap=True)
    print("Test metrics: " + str(metrics))

if 'viz' in datasets:
    print("Saving Visualizations")
    save_risk_surface_visualizations(model, datasets['viz'], norm_vals=norm_vals,
                                     output_dir=args.results_dir, plot_error=args.plot_error,
                                     experiment=args.experiment, trt_idx=args.treatment_idx)

if 'test' in datasets and args.treatment_idx is not None:
    print("Calculating treatment recommendation survival curvs")
    # We use the test dataset because these experiments don't have a viz dataset
    save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir,
                                      trt_idx=args.treatment_idx)

if args.results_dir:
    _, model_str = os.path.split(args.model)
    output_file = os.path.join(args.results_dir, "models") + model_str + str(uuid.uuid4()) + ".h5"
    print("Saving model parameters to output file", output_file)
    save_model(model, output_file)