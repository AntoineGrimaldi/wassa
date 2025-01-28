import torch
import numpy as np
import matplotlib.pyplot as plt
from wassa.wassa_plots import plot_results_std, plot_SM, plot_colored_raster
from wassa.dataset_generation import sm_generative_model, generate_dataset
from wassa.wassa_utils import performance_as_a_function_of_number_of_motifs

date = '2024_01_24'
#device = torch.device("cuda" if torch.cuda.is_available() else "CPU")
device = 'cuda:1'
print(device)

class dataset_parameters():
    seed = 666
    
    N_pre = 100 # number of neurons
    N_timesteps = 255 # number of timesteps for the raster plot (in ms)
    N_samples = 60 # total number of samples in the dataset

    N_delays = 51 # number of timesteps in spiking motifs, must be a odd number for convolutions
    N_SMs = 1 # number of structured spiking motifs
    N_involved = N_pre*torch.ones(N_SMs) # number of neurons involved in the spiking motif
    avg_fr = 20 # average firing rate of the neurons (in Hz)
    std_fr = .1 # standard deviation for the firing rates of the different neurons
    frs = torch.normal(avg_fr, std_fr, size=(N_pre,)).abs()
    freq_sms = 16*torch.ones(N_SMs) # frequency of apparition of the different spiking motifs (in Hz)
    overlapping_sms = False # possibility to have overlapping sequences

    temporal_jitter = .1 # temporal jitter for the spike generation in motifs
    dropout_proba = 0 # probabilistic participations of the different neurons to the spiking motif
    additive_noise = .1 # percentage of background noise/spontaneous activity
    warping_coef = 1 # coefficient for time warping

    def get_parameters(self):
        return f'{self.N_pre}_{self.N_delays}_{self.N_SMs}_{self.N_timesteps}_{self.N_samples}_{self.N_involved.mean()}_{self.avg_fr}_{self.freq_sms.mean()}_{self.overlapping_sms}_{self.temporal_jitter}_{self.dropout_proba}_{self.additive_noise}_{self.warping_coef}_{self.seed}'

class training_parameters:
    kernel_size = (dataset_parameters.N_SMs, dataset_parameters.N_pre, dataset_parameters.N_delays)
    loss_type = 'mse'
    N_learnsteps = 1000
    learning_rate = .001
    penalty_type = 'smoothed_orthogonality'
    smoothwind = 40
    lambda_ = .1
    batch_size = None
    output = 'linear' 
    do_bias = True 
    zeros = 'ignore'
    wass_order = 1
    weight_init = None
    if not penalty_type:
        lambda_ = 0
    elif penalty_type[:8] != 'smoothed': 
        smoothwind = 0
    if lambda_ == 0:
        penalty_type = None
    def get_parameters(self):
        name = f'{self.loss_type}_{self.output}_{self.penalty_type}_{self.do_bias}_{self.kernel_size}_{self.N_learnsteps}_{self.learning_rate}_{self.lambda_}_{self.batch_size}_{self.smoothwind}'
        if self.loss_type == 'emd':
            name += f'_{self.zeros}_{self.wass_order}'
        return name

params_mse = training_parameters()
params_emd = training_parameters()
params_emd.loss_type = 'emd'
params_emd.penalty_type = 'cc'
params_emd.lambda_ = 1

N_iter = 20
seeds = torch.arange(0,N_iter)
num_patterns = torch.arange(1,10)
results, num_patterns = performance_as_a_function_of_number_of_motifs(dataset_parameters, params_emd, params_mse, date, num_patterns, N_iter = N_iter, seeds = seeds, do_seqnmf = True, device=device)