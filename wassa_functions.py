import torch, os
from dataset_generation import generate_dataset, get_dataset_parameters, make_allen_dataset
from wassa_metrics import get_similarity, WassDist, kernels_diff
from wassa_training import learn_motifs, unsupervised_learning, get_training_parameters
from wassa_utils import in_notebook, correlation_kernels
from wassa import WassA
if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def performance_as_a_function_of_hp(file_name, training_parameters_emd, training_parameters_mse, hyperparameters, training_metrics, dataset, dataset_name = None, synthetic_metrics = [], seeds = False, n_iter = 5, kfold = 3, plot = False, verbose =  False, device='cpu'):

    if ~seeds:
        seeds = torch.arange(n_iter)
        
    if os.path.isfile(file_name):
        results, hp_grid, saved_hyperparameters = torch.load(file_name, map_location='cpu', weights_only = True)
        assert saved_hyperparameters == hyperparameters, print(saved_hyperparameters, hyperparameters)
    else:
        num_hp = len(hyperparameters)
        num_var = []
        for name, values in hyperparameters.items():
            num_var.append(len(values))
        hp_grid = torch.zeros(num_var)
        results = torch.zeros(num_var+[2,n_iter,kfold,len(training_metrics)+len(synthetic_metrics)])
        results[:] = torch.nan

    pbar = tqdm(total=len(torch.where(hp_grid==0)[0]))
    while len(torch.where(hp_grid==0)[0])>0:
        hp_loc_possible = torch.where(hp_grid==0)
        loc_indice = torch.randint(len(hp_loc_possible[0]), [1])
        ind_hp = 0
        list_indices = []
        for hp_name, hp_values in hyperparameters.items():
            indice = hp_loc_possible[ind_hp][loc_indice]
            list_indices.append([indice])
            training_parameters[hp_name] = hp_values[indice]
            ind_hp+=1
            
        params_mse = training_parameters_mse.copy()
        params_emd = training_parameters_emd.copy()
        
        if dataset_name is not None:
            assert synthetic_metrics == [], 'real dataset case but synthetic metrics'
            results[list_indices+[[0]]] = unsupervised_learning([params_mse],dataset,training_metrics,dataset_name, k_fold=kfold,n_iter=n_iter,verbose=verbose,device=device)
            results[list_indices+[[1]]] = unsupervised_learning([params_emd],dataset,training_metrics,dataset_name, k_fold=kfold,n_iter=n_iter,verbose=verbose,device=device)
        else:
            assert len(synthetic_metrics)>0, 'synthetic dataset case but no synthetic metrics'
            dataset_parameters = dataset.copy()
            for i in range(n_iter):
                dataset_parameters['seed'] = seeds[i]
                sm, dataset_input, dataset_output = generate_dataset(dataset_parameters, verbose=False, device=device)
                results[list_indices+[[0],[i]]] = unsupervised_learning([params_mse],[sm, dataset_input],training_metrics, get_dataset_parameters(dataset_parameters, include_samples=True),k_fold=kfold,synthetic_metrics=synthetic_metrics,verbose=verbose, device=device)
                results[list_indices+[[1],[i]]] = unsupervised_learning([params_emd],[sm, dataset_input],training_metrics, get_dataset_parameters(dataset_parameters, include_samples=True),k_fold=kfold,synthetic_metrics=synthetic_metrics,verbose=verbose, device=device)
                
        hp_grid[list_indices] = 1
        pbar.update(1)
        if os.path.isfile(file_name):
            saved_results, saved_hp_grid, _ = torch.load(file_name, map_location='cpu', weights_only = True)
            saved_results[list_indices] = results[list_indices]
            results = saved_results
            hp_grid += saved_hp_grid
            hp_grid[hp_grid>0] = 1
        torch.save([results, hp_grid, hyperparameters], file_name)
        
    pbar.close()
    return results, hp_grid

def performance_as_a_function_of_dataset_parameters(file_name, dataset_parameters, training_parameters_emd, training_parameters_mse, dataset_variables, training_metrics, synthetic_metrics, seeds = False, n_iter = 5, kfold = 3, do_seqnmf = False, plot = False, verbose =  False, device='cpu'):

    # TODO assertion on the keys of dataset_variables
    world_parameters = dataset_parameters.copy()
    
    if ~seeds: seeds = torch.arange(n_iter)
    if do_seqnmf: file_name+='_seqnmf'
        
    if os.path.isfile(file_name):
        results, hp_grid, saved_dataset_variables = torch.load(file_name, map_location='cpu', weights_only = True)
        assert saved_dataset_variables == dataset_variables
    else:
        num_hp = len(dataset_variables)
        num_var = []
        for name, values in dataset_variables.items():
            num_var.append(len(values))
        hp_grid = torch.zeros(num_var)
        if do_seqnmf:
            results = torch.zeros(num_var+[3,n_iter,kfold,len(training_metrics)+len(synthetic_metrics)])
        else:
            results = torch.zeros(num_var+[2,n_iter,kfold,len(training_metrics)+len(synthetic_metrics)])
        results[:] = torch.nan

    params_mse = training_parameters_mse.copy()
    params_emd = training_parameters_emd.copy()
    
    pbar = tqdm(total=len(torch.where(hp_grid==0)[0]))
    while len(torch.where(hp_grid==0)[0])>0:
        hp_loc_possible = torch.where(hp_grid==0)
        loc_indice = torch.randint(len(hp_loc_possible[0]), [1])
        ind_hp = 0
        list_indices = []
        for hp_name, hp_values in dataset_variables.items():
            indice = hp_loc_possible[ind_hp][loc_indice]
            list_indices.append([indice])
            world_parameters[hp_name] = hp_values[indice]
            if hp_name == 'N_sms':
                world_parameters.update({'N_involved': world_parameters['N_pre']*torch.ones(world_parameters['N_sms'])}) 
                world_parameters.update({'proba_sms' : torch.ones(world_parameters['N_sms'])/world_parameters['N_sms']})
                params_emd['kernel_size'] = (world_parameters['N_sms'],world_parameters['N_pre'],world_parameters['N_timesteps'])
                params_mse['kernel_size'] = (world_parameters['N_sms'],world_parameters['N_pre'],world_parameters['N_timesteps'])
            ind_hp+=1

        for i in range(n_iter):
            world_parameters['seed'] = seeds[i].item()
            sm, dataset_input, dataset_output = generate_dataset(world_parameters, verbose=False, device=device)
            results[list_indices+[[0],[i]]] = unsupervised_learning([params_mse],[sm, dataset_input.sum(axis=0)],training_metrics, get_dataset_parameters(world_parameters, include_samples=True),k_fold=kfold,synthetic_metrics=synthetic_metrics,verbose=verbose, device=device)
            results[list_indices+[[1],[i]]] = unsupervised_learning([params_emd],[sm, dataset_input.sum(axis=0)],training_metrics, get_dataset_parameters(world_parameters, include_samples=True),k_fold=kfold,synthetic_metrics=synthetic_metrics,verbose=verbose, device=device)
            if do_seqnmf:
                results[list_indices+[[3],[i]]] = compute_seqnmf(sm, dataset_input.sum(axis=0), params_mse, training_metrics, synthetic_metrics, k_fold = kfold)
                
        hp_grid[list_indices] = 1
        pbar.update(1)
        if os.path.isfile(file_name):
            saved_results, saved_hp_grid, _ = torch.load(file_name, map_location='cpu', weights_only = True)
            saved_results[list_indices] = results[list_indices]
            results = saved_results
            hp_grid += saved_hp_grid
            hp_grid[hp_grid>0] = 1
        torch.save([results, hp_grid, dataset_variables], file_name)
        
    pbar.close()
    return results, hp_grid

def allen_reconstruction_comparison(dataset_name, training_parameters, metric_names, frs=False, saving_path = 'results/allen_',device='cpu'):

    mse_loss = torch.nn.MSELoss()
    emd_loss = WassDist(zeros='same',normalize=training_parameters['normalize_input'])

    dataset_path = '../../'+dataset_name+'/*'
    all_trainsets, all_testsets, all_othersets = make_allen_dataset(dataset_path)
    results = torch.zeros([len(all_trainsets),7,3])
    
    for ind in tqdm(range(len(all_trainsets))):

        if frs:
            training_set, testing_set, othersets = all_trainsets[ind].mean(dim=-1).unsqueeze(-1).to(device), all_testsets[ind].mean(dim=-1).unsqueeze(-1).to(device), all_othersets[ind].mean(dim=-1).unsqueeze(-1).to(device)
        else:
            training_set, testing_set, othersets = all_trainsets[ind].to(device), all_testsets[ind].to(device), all_othersets[ind].to(device)

        num_samples, num_neurons, num_timesteps = training_set.shape
        training_parameters['kernel_size'] = (1,num_neurons, num_timesteps)
        model = WassA(training_parameters,device=device)
        model_name = saving_path+dataset_name+str(ind)+get_training_parameters(training_parameters)
        model, training_metrics, _ = learn_motifs(model,training_set,testing_set,training_parameters,model_name,metric_names,None,verbose=False)
        datasets = [training_set,testing_set,othersets]
        for ind_set, dataset in enumerate(datasets):
            if training_parameters['normalize_input']:
                dataset = torch.nn.functional.normalize(dataset, p=1, dim=2)
            factors, dataset_hat = model(dataset)
            results[ind,0,ind_set] = mse_loss(dataset_hat,dataset).detach()
            results[ind,1,ind_set] = emd_loss(dataset_hat,dataset).detach()
            results[ind,2,ind_set] = (1-torch.var(dataset-dataset_hat)/torch.var(dataset)).nanmean().detach()
            correlation_kernel, mse_kernel, emd_kernel = 0, 0, 0
            for i in range(dataset.shape[0]):
                input_spikes, weights = dataset[i].unsqueeze(0).detach(), model.weights.detach()
                correlation_kernel += correlation_kernels(input_spikes, weights)
                mse_kernel += kernels_diff(input_spikes, weights, 'mse')
                emd_kernel += kernels_diff(input_spikes, weights, 'emd')
            results[ind,3,ind_set] = correlation_kernel/dataset.shape[0]
            results[ind,4,ind_set] = mse_kernel/dataset.shape[0]
            results[ind,5,ind_set] = emd_kernel/othersets.shape[0]
            results[ind,6,ind_set] = factors.detach().mean()

    return results
