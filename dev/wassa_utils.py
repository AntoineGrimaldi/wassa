import torch

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

def sliding_correlation(x,x_hat=None,mode='max',p=2):
    
    n_samples, n_channels, n_time = x.shape
    padding_steps = torch.div(n_time,2,rounding_mode='floor')
    padded_x = torch.nn.functional.pad(x,(padding_steps,padding_steps,0,0,0,0), mode='constant')
    if x_hat is None:
        x_hat = x.clone()
    cross_correlation = torch.nn.functional.conv1d(padded_x,x_hat)
        
    if mode == 'mean':
        reduced_cc = cross_correlation.mean(dim=-1)
    elif mode == 'max':
        reduced_cc = cross_correlation.amax(dim=-1)

    if p=='max':
        norme = torch.amax(x, dim=(1,2)).unsqueeze(1).repeat(1,x_hat.shape[0])*torch.amax(x_hat, dim=(1,2)).unsqueeze(0).repeat(n_samples,1)
    else:
        norme = torch.norm(x,p=p,dim=(1,2)).unsqueeze(1).repeat(1,x_hat.shape[0])*torch.norm(x_hat,p=p,dim=(1,2)).unsqueeze(0).repeat(n_samples,1)
    output = reduced_cc/(norme+1e-14)
    output[output!=output] = 0
    return output

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

def cross_correlation_comp(factors,mode='mean'):
    n_batch, n_kernel, n_timesteps = factors.shape
    reshaped_factors = factors.reshape(n_batch*n_kernel,1,n_timesteps)
    cross_correlation_factors = sliding_correlation(reshaped_factors,mode=mode)
    blocks = torch.ones([n_kernel,n_kernel],device=factors.device).triu(diagonal=1)
    bloc_diag = torch.block_diag(*blocks.unsqueeze(0).repeat(n_batch,1,1))
    valid_cross_corr = cross_correlation_factors[bloc_diag==1]
    return torch.mean(valid_cross_corr)

#TODO check if it's the same as moving average
def smoothing(x,smoothing_window_size):
    n_batch, n_kernel, n_timesteps = x.shape
    weights = torch.ones([1,1,smoothing_window_size], device=x.device)/smoothing_window_size
    flattened = torch.reshape(x, (n_batch*n_kernel, 1, n_timesteps))
    smoothed = torch.nn.functional.conv1d(flattened, weights)
    return torch.reshape(smoothed, (n_batch, n_kernel, smoothed.shape[-1]))

def kernels_similarity(weights):
    max_cross_correlation = sliding_correlation(weights,mode='max')
    return max_cross_correlation[torch.triu_indices(weights.shape[0],weights.shape[0],offset=1).unbind()].mean()

def factors_similarity(factors):
    corrcoef = static_correlation(factors)
    return corrcoef[torch.triu_indices(corrcoef.shape[0],corrcoef.shape[0],offset=1).unbind()].abs().mean()

def correlation_latent_variables(true, learnt):
    corrcoef = static_correlation(true,learnt)
    return find_closest(corrcoef,'max')

def correlation_kernels(true, learnt, p=2):
    max_cross_correlation = sliding_correlation(true, learnt, mode='max', p=p)
    return find_closest(max_cross_correlation,'max')

def sparsity_kl(factors, rho):
    rho_hat = torch.mean(factors,dim=(0,2))
    epsilon = 1e-8
    rho_hat = torch.clamp(rho_hat, min=epsilon, max=1 - epsilon)
    return (rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))).mean()

def compute_factors_distributions(factors_trainset, factors_testset, bins = torch.arange(0,1.01,.01)):
    trainset_latent_variables, testset_latent_variables = factors_trainset.swapaxes(0,1).flatten(start_dim=1), factors_testset.swapaxes(0,1).flatten(start_dim=1)
    n_kernels = trainset_latent_variables.shape[0]
    trainset_latent_hist, testset_latent_hist = torch.zeros([n_kernels,bins.shape[0]-1]), torch.zeros([n_kernels,bins.shape[0]-1])
    for n in range(trainset_latent_variables.shape[0]):
        (trainset_latent_hist[n],_), (testset_latent_hist[n],_) = torch.histogram(trainset_latent_variables[n].detach().cpu(), bins), torch.histogram(testset_latent_variables[n].detach().cpu(), bins)
    return torch.nn.functional.normalize(trainset_latent_hist, p=1, eps=1e-8, dim=1), torch.nn.functional.normalize(testset_latent_hist, p=1, eps=1e-8, dim=1)

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
                W, H, cost, loadings, power = seqnmf(seqnmf_input, K=training_parameters['N_SMs'], L=training_parameters['N_delays'], Lambda=lambda_, max_iter=1000)
                torch.save([W, H, cost, loadings, power],path_seqnmf_)
    
            if torch.isnan(torch.tensor(W)).sum()==0:
                seqnmf_parameters = training_parameters.copy()
                seqnmf_parameters['tied_weights'] = True
                autoencoder = WassA(seqnmf_parameters,device=trainset.device)
                autoencoder.weights = torch.tensor(W,dtype=torch.float32).swapaxes(0,1)
                results[i,k,:] = get_similarity(sm, autoencoder, testset, synthetic_metrics)
    return results

