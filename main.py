import json
import h5py
import uuid
import lasagne
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import deepsurv
from deepsurv_logger import TensorboardLogger
from func import *


def dataset_to_df(dataset):
    covariates = pd.DataFrame(dataset['x'])
    time = pd.DataFrame(dataset['t'], columns=['time'])
    status = pd.DataFrame(dataset['e'], columns=['status'])
    return pd.concat([time, status, covariates], axis=1)


# load dataset
filename = "data/whas/whas_train_test.h5"
datasets = defaultdict(dict)
with h5py.File(filename, 'r') as fp:
    for ds in fp:
        for array in fp[ds]:
            datasets[ds][array] = fp[ds][array][:]

train_dataset = datasets['train']
# covariates = pd.DataFrame(train_dataset['x'])
# time = pd.DataFrame(train_dataset['t'], columns=['time'])
# status = pd.DataFrame(train_dataset['e'], columns=['status'])
# train_df = pd.concat([time, status, covariates], axis=1)
train_df = dataset_to_df(train_dataset)

test_dataset = datasets['test']
# covariates = pd.DataFrame(test_dataset['x'])
# time = pd.DataFrame(test_dataset['t'], columns=['time'])
# status = pd.DataFrame(test_dataset['e'], columns=['status'])
# test_df = pd.concat([time, status, covariates], axis=1)
test_df = dataset_to_df(test_dataset)

# # Extract the event and time columns as numpy arrays
# e = df['status'].values.astype(np.int32)
# t = df['time'].values.astype(np.float32)
#
# # Extract the patient's covariates as a numpy array
# x_df = df.drop(['status', 'time'], axis=1)
# x = x_df.values.astype(np.float32)
#
# # Return the DeepSurv dataframe
# train_data = {'x': x, 'e': e, 't': t}
train_data = dataframe_to_deepsurv_ds(train_df, event_col='status', time_col='time')

norm_vals = {
    'mean': datasets['train']['x'].mean(axis=0),
    'std': datasets['train']['x'].std(axis=0)
}

# Set up hyper-parameters
hyperparams = {
    "L2_reg": 2.364680908203125,
    "batch_norm": False,
    "dropout": 0.017243652343750002,
    "hidden_layers_sizes": [26, 26, 26],
    "learning_rate": 0.023094096518941305,
    "lr_decay": 0.0009819482421875,
    "momentum": 0.926554443359375,
    "n_in": 6,
    "standardize": True
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
update_fn = lasagne.updates.adam  # The type of optimizer to use
# Check out http://lasagne.readthedocs.io/en/latest/modules/updates.html for other optimizers to use
n_epochs = 1200
# If you have validation data, you can add it as the second parameter to the function
metrics = model.train(train_data, n_epochs=n_epochs, logger=logger, update_fn=update_fn)

# Print the final metrics
print('Train C-Index:', metrics['c-index'][-1])
# print('Valid C-Index: ',metrics['valid_c-index'][-1])

# plot the training / validation curves
visualize.plot_log(metrics)
plt.show()

# Evaluate model with training dataset
with open("./models/whas_model_selu_revision.0.json", 'r') as fp:
    json_model = fp.read()
    hyperparams = json.loads(json_model)

train_data = datasets['train']
if hyperparams['standardize']:
    train_data = utils.standardize_dataset(train_data, norm_vals['mean'], norm_vals['std'])

metrics = evaluate_model(model, train_data)
print("Training metrics: " + str(metrics))

# Evaluate model with validation set
# valid_data = datasets['valid']
# if hyperparams['standardize']:
#     valid_data = utils.standardize_dataset(valid_data, norm_vals['mean'], norm_vals['std'])
#     metrics = evaluate_model(model, valid_data)
# print("Valid metrics: " + str(metrics))

# Evaluate model with testing dataset
test_data = utils.standardize_dataset(datasets['test'], norm_vals['mean'], norm_vals['std'])
metrics = evaluate_model(model, test_data, bootstrap=True)
print("Test metrics: " + str(metrics))

# Save model weights
results_dir = "./results/"
_, model_str = os.path.split("./models/whas_model_selu_revision.0.json")
output_file = os.path.join(results_dir, "models") + model_str + str(uuid.uuid4()) + ".h5"
print("Saving model parameters to output file", output_file)
save_model(model, output_file)

# Save the hazard ratio prediction
hr_pred_train = pd.DataFrame(np.squeeze(model.predict_risk(train_data['x'])), columns=['hr'])
hr_pred_test = pd.DataFrame(np.squeeze(model.predict_risk(test_data['x'])), columns=['hr'])
train_hr_df = pd.concat([train_df, hr_pred_train], axis=1)
test_hr_df = pd.concat([test_df, hr_pred_test], axis=1)
# Save to excel files
print("Saving prediction results to excel files!")
train_hr_df.to_excel("./results/train_hr_pred_{}.xlsx".format(str(uuid.uuid4())))
test_hr_df.to_excel("./results/test_hr_pred_{}.xlsx".format(str(uuid.uuid4())))

# CMD [ "python", "-u", "/scripts/deepsurv_run.py", "whas", \
# "/models/whas_model_selu_revision.0.json", \
# "/shared/data/whas_train_test.h5", \
# "--update_fn", "adam", \
# "--results_dir", "/shared/results/", \
# "--num_epochs", "1200"]

# Risk surface plot with two covariates
# model.plot_risk_surface(test_data['x'])
# save_risk_surface_visualizations(model, datasets['viz'], norm_vals=norm_vals, output_dir=results_dir,
#                                  plot_error="store_true", experiment="whas", trt_idx=None)

# if 'test' in datasets and args.treatment_idx is not None:
#     print("Calculating treatment recommendation survival curves")
#     # We use the test dataset because these experiments don't have a viz dataset
#     save_treatment_rec_visualizations(model, test_dataset, output_dir=args.results_dir, trt_idx=None)
