# import sys
# sys.path.append('/deepsurv')
import lasagne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import deep_surv
import viz
from deepsurv_logger import TensorboardLogger


def dataframe_to_deepsurv_ds(df, event_col='Event', time_col='Time'):
    # Extract the event and time columns as numpy arrays
    e = df[event_col].values.astype(np.int32)
    t = df[time_col].values.astype(np.float32)

    # Extract the patient's covariates as a numpy array
    x_df = df.drop([event_col, time_col], axis=1)
    x = x_df.values.astype(np.float32)

    # Return the DeepSurv dataframe
    return {'x': x, 'e': e, 't': t}


# Read in dataset and print the first five elements to get a sense of what the dataset looks like
train_dataset_fp = 'example_data.csv'
train_df = pd.read_csv(train_dataset_fp)
# train_df.head()

# You can also use this function on your training dataset, validation dataset, and testing dataset
train_data = dataframe_to_deepsurv_ds(train_df, event_col='Event', time_col='Time')

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
model = deep_surv.DeepSurv(**hyperparams)

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
viz.plot_log(metrics)
plt.show()
