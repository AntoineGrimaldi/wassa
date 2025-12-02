import torch
import numpy as np
from scipy.optimize import curve_fit

def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config: 
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

def get_best_results(results, losses, metrics, saved_hyperparameters, indices=None, results_allen=False):

    max_mean_ind = []
    hyperparameters_names = list(saved_hyperparameters.keys())

    if results_allen:
        results = results.nanmean(axis=-2).swapaxes(-1,-2)

    if not indices:
        indices = []
        for i in range(len(results.shape)-3):
            if results.shape[i]>1:
                indices.append(i)
    for ind_l, loss in enumerate(losses):
        print(f'====== {loss} ======')
        for ind_m, metric in enumerate(metrics):
            results_l = results[...,ind_l,ind_m,:]
            mean_results = torch.nanmean(results_l,dim=-1)
            if metric in ['training loss', 'testing loss', 'explained variance', 'kernels correlation', 'train/test factors difference', 'mse', 'emd', 'emd mean timings']:
                mean_results[mean_results!=mean_results] = float('+inf')
                max_mean_ind_l = torch.where(mean_results.abs()==mean_results.abs().min())
            elif metric in ['factors similarity', 'kernels similarity', 'mean timings similarity']:
                mean_results[mean_results!=mean_results] = float('-inf')
                max_mean_ind_l = torch.where(mean_results==mean_results.max())
            else:
                print('wrong metric name')
            if max_mean_ind_l[0].shape[0]>1:
                print(f'{max_mean_ind_l[0].shape[0]} identical results')
                max_mean_ind_l = [max_mean_ind_l[k][0] for k in range(len(max_mean_ind_l))]
            max_mean_ind.append(max_mean_ind_l)
            hyp_max_mean = [f'{hyperparameters_names[k]} : {saved_hyperparameters[hyperparameters_names[k]][max_mean_ind_l[k]]}' for k in indices]

            print(f'Best mean for {metric} : {mean_results[max_mean_ind_l].item()}\n{hyp_max_mean}')

# TODO: check if it can be assimilated to sliding_correlation with padding = 0
def static_correlation(x,x_hat=None,normalize=True):
    n_batch, n_variables, n_timesteps = x.shape
    if x_hat is None:
        x_hat = x.clone()
    _, n_variables_hat, _ = x_hat.shape
    if normalize:
        x = torch.nn.functional.normalize(x, p=2, eps=1e-14, dim=2)
        x_hat = torch.nn.functional.normalize(x_hat, p=2, eps=1e-14, dim=2)
    shaped_x = x.swapaxes(0,1).flatten(start_dim=1).unsqueeze(1).repeat(1,n_variables_hat,1)
    shaped_x_hat = x_hat.swapaxes(0,1).flatten(start_dim=1).unsqueeze(0).repeat(n_variables,1,1)
    mean_x = shaped_x.mean(axis=-1).unsqueeze(-1).repeat(1,1,n_batch*n_timesteps)
    mean_x_hat = shaped_x_hat.mean(axis=-1).unsqueeze(-1).repeat(1,1,n_batch*n_timesteps)
    var_x = ((shaped_x-mean_x)**2).mean(axis=-1)
    var_x_hat = ((shaped_x_hat-mean_x_hat)**2).mean(axis=-1)
    covar = ((shaped_x-mean_x)*(shaped_x_hat-mean_x_hat)).mean(axis=-1)
    return covar/(torch.sqrt(var_x*var_x_hat)+1e-14)

def find_closest(matrix,max_or_min):
    # size of the matrix is MxN where N is the number of trained kernels and M is the number of ground truth motifs
    # loop over the different motifs to see if similarity is good with different kernels 
    value = 0
    for l_k in range(min(matrix.shape)):
        if max_or_min=='max':
            ind = torch.where(matrix==matrix.max())
        elif max_or_min=='min':
            ind = torch.where(matrix==matrix.min())
        ind_kernel, ind_gm = ind
        value += matrix[ind_kernel[0], ind_gm[0]]
        if max_or_min=='max':
            matrix[:,ind_gm[0]] = -1e14
        elif max_or_min=='min':
            matrix[:,ind_gm[0]] = 1e14
    return value/matrix.shape[0]

def kernels_similarity(weights):
    max_cross_correlation = static_correlation(weights.swapaxes(0,1))
    return max_cross_correlation[torch.triu_indices(weights.shape[0],weights.shape[0],offset=1).unbind()].abs().mean()

def factors_similarity(factors):
    corrcoef = static_correlation(factors,normalize=False)
    return corrcoef[torch.triu_indices(corrcoef.shape[0],corrcoef.shape[0],offset=1).unbind()].abs().mean()

def correlation_kernels(true, learnt):
    cross_correlation = static_correlation(true.swapaxes(0,1), learnt.swapaxes(0,1))
    return find_closest(cross_correlation,'max')

def correlation_mean_timings(true,learnt,norm=1):
    if norm == 1:
        learnt = torch.nn.functional.normalize(learnt, p=1, dim=2)
    elif norm == 2:
        learnt = torch.nn.functional.normalize(learnt, p=2, dim=(1,2))
    elif norm == 'max':
        max_val = torch.amax(learnt, dim=(1,2)).unsqueeze(1).unsqueeze(2).repeat(1,learnt.shape[1],learnt.shape[2])
        learnt.div_(max_val)
    return (true*learnt).sum()/true.sum()


def compute_seqnmf(sm, dataset, training_parameters, metric_names, synthetic_metrics, lambda_ = 0, n_iter = 1, k_fold = 1):
    results = torch.zeros([n_iter,k_fold,len(metric_names)+len(synthetic_metrics)])
    results[:] = torch.nan
    trainsets, testsets, indices = kfold_dataset(dataset.to(torch.float).to(device),k=k_fold)
    for i in range(n_iter):
        for k in range(k_fold):
            path_seqnmf_ = f'../results/{date}_seqnmf_{get_dataset_parameters(sm.opt)}_{lambda_}' + str(i) + str(k)
            trainset, testset = trainsets[k], testsets[k]
            if os.path.isfile(file_name):
                W, H, cost, loadings, power = torch.load(file_name, weights_only = True)
            else:
                seqnmf_input = torch.cat([trainset[i] for i in range(trainset.shape[0])],dim=-1).to('cpu')
                W, H, cost, loadings, power = seqnmf(seqnmf_input, K=training_parameters['N_sms'], L=training_parameters['N_delays'], Lambda=lambda_, max_iter=1000)
                torch.save([W, H, cost, loadings, power],path_seqnmf_)
    
            if torch.isnan(torch.tensor(W)).sum()==0:
                seqnmf_parameters = training_parameters.copy()
                seqnmf_parameters['tied_weights'] = True
                autoencoder = WassA(seqnmf_parameters,device=trainset.device)
                autoencoder.weights = torch.tensor(W,dtype=torch.float32).swapaxes(0,1)
                results[i,k,:] = get_similarity(sm, autoencoder, testset, synthetic_metrics)
    return results

def gaussian_kernel(x, amplitude, mean, std):
    return amplitude * np.exp(-((x - mean) / 4 / std)**2)

def estimate_spike_times(all_weights, dataset_parameters = None, max_iteration = 5000, min_max_proba = 1e-5):

    n_sms, n_neurons, n_timesteps = all_weights.shape
    all_weights = all_weights / (all_weights.sum(axis=-1,keepdims=True)+1e-14).repeat(n_timesteps, axis=-1)
    time_scale = np.arange(n_timesteps)
    all_spike_times = []
    all_errors = []
    for sm in range(n_sms):
        spike_times = []
        errors = []
        for neuron in range(n_neurons):
            if dataset_parameters is not None:
                jitter = dataset_parameters['temporal_jitter']
            else:
                jitter = .01
            max_proba = all_weights[sm,neuron].max()
            time = np.where(all_weights[sm,neuron]==max_proba)[0][0]
            jitter = 1/(max_proba*np.sqrt(2*np.pi))
            if max_proba < min_max_proba:
                spike_times.append(n_timesteps//2)
                errors.append(np.inf)
            elif max_proba>.99:
                spike_times.append(round(time))
                errors.append(0)
            else:
                popt, _ = curve_fit(gaussian_kernel, time_scale, all_weights[sm,neuron], p0=[max_proba,time,jitter], bounds = ((0,0,0), (1,n_timesteps, 1/(min_max_proba*np.sqrt(2*np.pi)))), maxfev=max_iteration)
                error = ((all_weights[sm,neuron]-gaussian_kernel(time_scale,popt[0],popt[1],popt[2]))**2).mean()
                spike_times.append(round(popt[1]))
                errors.append(error)
        all_spike_times.append(spike_times)
        all_errors.append(errors)
    return all_spike_times, all_errors