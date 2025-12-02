import torch, os, time
from dataset_generation import kfold_dataset
from wassa import WassA
from wassa_metrics import WassDist, get_similarity
from wassa_utils import kernels_similarity, factors_similarity, in_notebook, estimate_spike_times, correlation_kernels, correlation_mean_timings
import adamw_eg
from wassa_plots import online_monitoring
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def learn_motifs(model,training_set,testing_set,training_parameters,path,metric_names,world,online_plots=False,verbose=True):
    
    if os.path.isfile(path + '.pth'):
        if verbose: print(path)
        model.load_state_dict(torch.load(path + '.pth', map_location=torch.device(training_set.device),weights_only=True))
        training_metrics,loaded_metric_names = torch.load(path + '_loss.pth',weights_only=True)
        if len(loaded_metric_names) > len(metric_names):
            indexes = []
            for name in metric_names:
                indexes.append(loaded_metric_names.index(name))
            training_metrics = training_metrics[:,indexes]
        elif len(loaded_metric_names) < len(metric_names):
            print(f' saved metrics : {loaded_metric_names}')
    else:
        
        if training_parameters['loss_type'] == 'mse':
            criterion = torch.nn.MSELoss()
        elif training_parameters['loss_type'] == 'emd':
            criterion = WassDist(zeros='same',normalize=training_parameters['normalize_input'])
        elif training_parameters['loss_type'] == 'bce':
            criterion = torch.nn.BCELoss()
        
        training_metrics = torch.zeros([training_parameters['N_learnsteps'],len(metric_names)])
        optimizer = adamw_eg.AdamWeg(model.parameters(), lr=training_parameters['learning_rate'])
        #optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters['learning_rate'])
        if online_plots: start_time = time.time()
        if training_parameters['batch_size']:
            nb_batch = torch.div(training_set.shape[0],training_parameters['batch_size'],rounding_mode='floor')
            if nb_batch*training_parameters['batch_size']!=training_set.shape[0]: nb_batch+=1
        else:
            nb_batch = 1

        for i_step in (tqdm(range(training_parameters['N_learnsteps'])) if verbose else range(training_parameters['N_learnsteps'])):

            # shuffle sample indices
            idx = torch.randperm(training_set.size(0))
            training_set = training_set[idx]
            
            for n in range(nb_batch):
                if training_parameters['batch_size']:
                    X = training_set[n*training_parameters['batch_size']:min((n+1)*training_parameters['batch_size'],training_set.shape[0])]
                else:
                    X = training_set
                if training_parameters['normalize_input']:
                    X = torch.nn.functional.normalize(X, p=1, dim=2)
                    testing_set = torch.nn.functional.normalize(testing_set, p=1, dim=2)
                optimizer.zero_grad()
                
                factors_train, reconstruction = model(X)
                loss = criterion(reconstruction,X)
                penalty_value = penalty(model,factors_train,training_parameters)
                loss += penalty_value
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.normalization()
                training_metrics[i_step] = compute_training_metrics(world, metric_names, model, criterion, testing_set, loss, penalty_value, factors_train, training_parameters)
                if online_plots and i_step%10==0:
                    online_monitoring(model.weights.detach().cpu(),training_metrics,metric_names,i_step,start_time)

        torch.save(model.state_dict(), path + '.pth')
        torch.save([training_metrics,metric_names], path + '_loss.pth')

    return model, training_metrics, metric_names

def penalty(model,factors,training_parameters):
    penalty = 0
    if 'factors_cc' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('factors_cc')]
        penalty += lambda_*factors_similarity(factors)
    if 'kernels_orthogonality' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('kernels_orthogonality')]
        penalty += lambda_*kernels_similarity(model.weights)
    if 'kernels_cumsum_orthogonality' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('kernels_cumsum_orthogonality')]
        penalty += lambda_*kernels_similarity(model.weights.cumsum(dim=-1))
    if 'sparsity_factors' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('sparsity_factors')]
        penalty += lambda_*factors.mean()
    if 'sparsity_weights' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('sparsity_weights')]
        penalty += lambda_*model.weights.mean()
    if 'bias' in training_parameters['penalty_type']:
        lambda_ = training_parameters['lambda'][training_parameters['penalty_type'].index('bias')]
        penalty += lambda_*model.bias.mean()
    return penalty

def compute_training_metrics(sm, metric_names, model, criterion, testing_set, loss, penalty_value, factors_train, training_parameters):
    learnt_weights = model.weights.detach()
    training_metrics = torch.zeros([len(metric_names)])
    training_metrics[:] = torch.nan
    if len(set(metric_names).intersection(['testing loss','explained variance','train/test factors difference']))>0:
        factors_test, testset_hat = model(testing_set)
    for ind_m, metric in enumerate(metric_names):
        if metric == 'training loss':
            training_metrics[ind_m] = (loss-penalty_value).detach()
        if metric == 'penalty value':
            training_metrics[ind_m] = penalty_value.detach()
        if metric == 'testing loss':
            training_metrics[ind_m] = criterion(testset_hat,testing_set).detach()
        if metric == 'explained variance':
            ev = 1-torch.var(testing_set-testset_hat)/torch.var(testing_set)
            training_metrics[ind_m] = ev.nanmean().detach()
        if metric == 'kernels correlation':
            training_metrics[ind_m] = kernels_similarity(learnt_weights)
        if metric == 'kernels similarity':
            training_metrics[ind_m] = correlation_kernels(sm.SMs, learnt_weights)
        if metric == 'mean timings similarity':
            mean_timings = torch.zeros_like(learnt_weights, device=testing_set.device)
            for ind_motif, spike_times_motif in enumerate(sm.spike_times):
                for ind, spike_addresses in enumerate(spike_times_motif):
                    mean_timings[ind_motif][spike_addresses] = 1
            training_metrics[ind_m] = correlation_mean_timings(mean_timings, learnt_weights)
        if metric == 'activations':
            training_metrics[ind_m] = factors_train.mean()
    return training_metrics

def unsupervised_learning(parameters,dataset,metric_names,dataset_name,k_fold=1,n_iter=1,last_steps=.05,synthetic_metrics=[],record_path='../simulations/',device='cpu',verbose=False):
    assertion_check(parameters[0],metric_names,synthetic_metrics)
    if len(synthetic_metrics)==0:
        n_samples, n_neurons, n_timesteps = dataset.shape
        sm = None
    else:
        sm, dataset = dataset
    trainsets, testsets, indices = kfold_dataset(dataset.to(torch.float).to(device),k=max(2,k_fold))
    results = torch.zeros([n_iter,k_fold,len(metric_names)+len(synthetic_metrics)])
    results[:] = torch.nan
    if verbose : pbar = tqdm(total=n_iter*k_fold)
    for i in range(n_iter):
        for k in range(k_fold):
            model_name = record_path + dataset_name + str(i) + str(k)
            trainset, testset = trainsets[k], testsets[k]
            if len(parameters)>1:
                learnsteps_init = parameters[0]['N_learnsteps']
                for params in parameters:
                    params.update({'N_learnsteps' : learnsteps_init//len(parameters)})
                    model_name += get_training_parameters(params)
                    autoencoder = WassA(params, device=device)
                    autoencoder, training_metrics, metric_names = learn_motifs(autoencoder,trainset,testset,params,model_name,metric_names,sm,verbose=verbose)
                    params.update({'N_learnsteps' : learnsteps_init})
            else:
                params = parameters[0]
                model_name += get_training_parameters(params)
                autoencoder = WassA(params, device=device)
                autoencoder, training_metrics, metric_names = learn_motifs(autoencoder,trainset,testset,params,model_name,metric_names,sm,verbose=verbose)
            
            nan_condition = torch.isnan(autoencoder.weights.data).sum()==0
            if 'mean time diff' in synthetic_metrics:
                spike_times, errors = estimate_spike_times(autoencoder.weights.detach().cpu().numpy())
            else:
                spike_times = None
            
            if nan_condition:
                n_last_steps = int(last_steps*parameters[0]['N_learnsteps'])
                if len(synthetic_metrics)==0:
                    results[i,k,:] = training_metrics[n_last_steps:].mean(axis=0)
                else:
                    similarities_gt = get_similarity(sm, autoencoder, testset, synthetic_metrics, spike_times = spike_times)
                    results[i,k] = torch.hstack([training_metrics[n_last_steps:].mean(axis=0),similarities_gt])
            else:
                print('nans in weights')
                print(model_name)
                
            if verbose : pbar.update(1)
    if verbose : pbar.close()
    return results

def assertion_check(training_parameters,metric_names,synthetic_metrics):
    assert len(training_parameters['penalty_type']) == len(training_parameters['lambda']), 'Different penalty types and amount of parameters'
    assert len(set(metric_names).intersection(['training loss','testing loss','penalty value','explained variance','kernels correlation']))==len(metric_names), 'Check training metrics name'
    assert len(set(training_parameters['penalty_type']).intersection([None,'factors_cc','kernels_cumsum_orthogonality','kernels_orthogonality','sparsity_factors','sparsity_weights','bias']))==len(training_parameters['penalty_type']), 'Check penalty name'
    assert len(set(synthetic_metrics).intersection(['factors similarity', 'kernels similarity', 'mean timings similarity', 'mean time diff', 'mse', 'emd', 'emd mean timings']))==len(synthetic_metrics), 'Check synthetic similarity metrics name'

def get_training_parameters(training_parameters):

    lambdaz = training_parameters['lambda'].copy()
    penaltiz = training_parameters['penalty_type'].copy()

    if 0 in lambdaz:
        penaltiz[lambdaz.index(0)] = None
    if None in penaltiz:
        lambdaz[penaltiz.index(None)] = 0
    if not 'smoothed_orthogonality' in penaltiz:
        training_parameters.update({'smoothwind' : 0})
    if not 'sparsity_factors' in penaltiz:
        training_parameters.update({'expected_patterns_per_sample' : 0})
            
    name = ''
    for hp_name, hp_value in training_parameters.items():
        if hp_name=='penalty_type':
            for penalty in penaltiz:
                name += f'{penalty}'
        elif hp_name=='lambda':
            for lambda_ in lambdaz:
                name += f'{lambda_}'
        elif isinstance(hp_value, tuple) or isinstance(hp_value, list):
            for value in hp_value:
                name += f'{value}'
        else:
            name += f'{hp_value}'
    return name