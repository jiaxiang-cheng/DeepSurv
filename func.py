import os
import sys
import time
import numpy as np

import visualize
import utils

localtime = time.localtime()
TIMESTRING = time.strftime("%m%d%Y%M", localtime)
# matplotlib.use('Agg')
sys.path.append("/DeepSurv/deepsurv")


def evaluate_model(model, dataset, bootstrap=False):
    def mse(model):
        def deepsurv_mse(x, hr, **kwargs):
            hr_pred = np.squeeze(model.predict_risk(x))
            return ((hr_pred - hr) ** 2).mean()

        return deepsurv_mse

    metrics = {'c_index': model.get_concordance_index(**dataset)}

    # calculate c-index
    if bootstrap:
        metrics['c_index_bootstrap'] = utils.bootstrap_metric(model.get_concordance_index, dataset)

    # calculate MSE
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


def save_risk_surface_visualizations(model, dataset, norm_vals, output_dir, plot_error, experiment, trt_idx):
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


def save_treatment_rec_visualizations(model, dataset, output_dir, trt_i=1, trt_j=0, trt_idx=0):
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
