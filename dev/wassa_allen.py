import torch
from wassa_functions import performance_as_a_function_of_hp

device = 'cuda:1'

saving_path = '../results/'

metric_names = ['training loss', 'testing loss', 'explained variance', 'kernels correlation', 'train/test factors difference']
synthetic_metrics = ['factors similarity', 'kernels similarity', 'mean timings similarity', 'mse', 'emd', 'emd mean timings']
number_iterations, number_folds = 2, 4

dataset_path = '../allen_data/'
dataset_name = 'bin_tensor_session_00_NM1_trunc'
data = torch.load(dataset_path+dataset_name+'.pt', weights_only = True)
avg_fr = torch.mean(data.to(torch.float),axis=(0,2))*1e3
n_samples, n_neurons, n_timesteps =  data.shape
avg_fr[avg_fr==0] = .1

dataset_parameters = {
    'seed' : 666,
    
    'N_pre' : n_neurons, # number of neurons
    'N_timesteps' : n_timesteps, # number of timesteps for the raster plot (in ms)
    'N_samples' : n_samples, # total number of samples in the dataset

    'N_delays' : 100, # number of timesteps in spiking motifs, must be a odd number for convolutions
    'N_SMs' : 5, # number of structured spiking motifs
    'avg_fr' : 20, # average firing rate of the neurons (in Hz)
    'std_fr' : .1, # standard deviation for the firing rates of the different neurons
    'overlapping_sms' : False, # possibility to have overlapping sequences

    'temporal_jitter' : 1, # temporal jitter for the spike generation in motifs
    'dropout_proba' : .1, # probabilistic participations of the different neurons to the spiking motif
    'additive_noise' : .1, # percentage of background noise/spontaneous activity
    'warping_coef' : 1, # coefficient for time warping

}
dataset_parameters.update({'N_involved':dataset_parameters['N_pre']*torch.ones(dataset_parameters['N_SMs'],device=device)}) # number of neurons involved in the spiking motif
dataset_parameters.update({'frs' : avg_fr.to(device)})
dataset_parameters.update({'freq_sms' : 5*torch.ones(dataset_parameters['N_SMs'],device=device)}) # frequency of apparition of the different spiking motifs (in Hz)
dataset_parameters.update({'coefficient_variation' : 1*torch.ones(dataset_parameters['N_pre'],device=device)}) # coefficient of variation of the FRs of each neuron (within VS. without a motif)

training_parameters = {
    'kernel_size' : (dataset_parameters['N_SMs'], dataset_parameters['N_pre'], dataset_parameters['N_delays']),
    'loss_type' : 'mse',
    'N_learnsteps' : 5000,
    'learning_rate' : .002,
    'beta_1' : .9,
    'penalty_type' : [None],
    'smoothwind' : 0,
    'expected_patterns_per_sample' : dataset_parameters['N_timesteps']*dataset_parameters['freq_sms'][0]*1e-3,
    'lambda' : [0],
    'batch_size' : None,
    'do_bias' : [False,False,False],
    'tied_weights' : False,
    'weight_init' : None,
    'normalize_input' : False,
    'reshape_input': False}

hyperparameters = {
'loss_type' : ['emd', 'mse'],
'learning_rate' : [.001, .005, .01, .015, .02, .05],
'penalty_type' : [['cc'], ['sparsity_factors'], ['kernels_orthogonality'], ['smoothed_orthogonality']],
'lambda' : [[.0001],[.001],[.01], [.1], [1], [10], [100]],
'normalize_input' : [False, True],
'reshape_input' : [True, False],
}

file_name = saving_path+'2025-08-25_hp_tuning_synthetic_'+dataset_name
results, hp_grid = performance_as_a_function_of_hp(file_name, training_parameters, hyperparameters, metric_names, dataset_parameters, synthetic_metrics = synthetic_metrics, n_iter = number_iterations, kfold = number_folds, device=device)
