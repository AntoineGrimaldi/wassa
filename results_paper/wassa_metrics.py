import torch
import matplotlib.pyplot as plt
import numpy as np
from wassa_utils import find_closest, correlation_kernels, correlation_mean_timings
from dataset_generation import gaussian_kernel

def torch_cdf_loss(reconstructed,input_seq,zeros,normalize):
    ''' Computes the 1D-Wasserstein distance along the last dimension of reconstructed and input_seq
        input_seq and reconstructed : BxNxT tensor where B is the number of batches, N the number of 
                                      neurons (or channels of the input) and T the number of timesteps
        zeros : treat neurons with no spike as informative ('same') or not ('ignore') -> nan
    '''
    assert reconstructed.shape == input_seq.shape
    
    n_timesteps = reconstructed.shape[-1]
    if normalize:
        reconstructed = torch.nn.functional.normalize(reconstructed, p=1, dim=2)
        input_seq = torch.nn.functional.normalize(input_seq, p=1, dim=2)
        if zeros == 'same':
            reconstructed[reconstructed.sum(dim=-1)==0], input_seq[input_seq.sum(dim=-1)==0] = 1/n_timesteps, 1/n_timesteps
            
    if zeros == 'ignore':
        cdf_reconstructed = torch.cumsum(reconstructed[input_seq.sum(dim=-1)>0],dim=-1)
        cdf_input = torch.cumsum(input_seq[input_seq.sum(dim=-1)>0],dim=-1)
    else:
        cdf_reconstructed = torch.cumsum(reconstructed,dim=-1)
        cdf_input = torch.cumsum(input_seq,dim=-1)

    return torch.abs(cdf_input-cdf_reconstructed)

class WassDist(torch.nn.Module):
    def __init__(self,zeros='ignore',normalize=True,reduction='mean'):
        super().__init__()
        self.zeros = zeros
        self.normalize = normalize
        self.reduction = reduction

    def forward(self, target, input_seq):
        if self.reduction == 'mean':
            wassa = torch_cdf_loss(target,input_seq,self.zeros,self.normalize).nanmean()
        elif self.reduction == 'sum':
            wassa = torch_cdf_loss(target,input_seq,self.zeros,self.normalize).nansum()
        elif self.reduction == 'none' and self.zeros == 'same':
            wassa = torch_cdf_loss(target,input_seq,self.zeros,self.normalize)
        else:
            print('zeros cannot be ignored if reduction is none')
        return wassa

def kernels_diff(true_kernels, learnt_kernels, metric, norm = None):

    true_kernels, learnt_kernels = true_kernels.clone(), learnt_kernels.clone()
    n_motifs, n_neurons, n_timbin = true_kernels.shape
    n_kernels = learnt_kernels.shape[0]

    if norm == 1:
        true_kernels = torch.nn.functional.normalize(true_kernels, p=1, dim=2)
        learnt_kernels = torch.nn.functional.normalize(learnt_kernels, p=1, dim=2)
    if norm == 2:
        true_kernels = torch.nn.functional.normalize(true_kernels, p=2, dim=(1,2))
        learnt_kernels = torch.nn.functional.normalize(learnt_kernels, p=2, dim=(1,2))

    if metric == 'mse':
        true_matrix = true_kernels.flatten(start_dim=1).unsqueeze(0).repeat(n_kernels,1,1)
        learnt_matrix = learnt_kernels.flatten(start_dim=1).unsqueeze(1).repeat(1,n_motifs,1)
        error_matrix = ((true_matrix-learnt_matrix)**2).mean(axis=-1)
    elif metric == 'emd':
        emd_loss = WassDist(zeros='same',normalize=True,reduction='none')
        true_matrix = true_kernels.unsqueeze(0).repeat(n_kernels,1,1,1)
        learnt_matrix = learnt_kernels.unsqueeze(1).repeat(1,n_motifs,1,1)
        error_matrix = emd_loss(true_matrix,learnt_matrix).mean(axis=(1,2))

    return find_closest(error_matrix,'min')

def normalize_times(kernel):

    if isinstance(kernel[0], tuple):
        times = np.array([t for (_, t) in kernel])
    else:
        times = np.array(kernel)

    tmin = times.min()
    tmax = times.max()
    return (times - tmin) / (tmax - tmin)

def spike_times_diff(true_spike_times, estimated_spike_times, min_warping_coef = 1):
    n_motifs, n_kernels = len(true_spike_times), len(estimated_spike_times)
    error_matrix = torch.inf*torch.ones([n_kernels,n_motifs])
    for m in range(n_motifs):
        for k in range(n_kernels):
            diff = 0
            for ind in range(len(true_spike_times[m])):
                true_neuron, true_time = true_spike_times[m][ind]
                estimated_time = estimated_spike_times[k][true_neuron]
                diff += abs(true_time-estimated_time)
            error_matrix[k,m] = diff/len(true_spike_times[m])
    return find_closest(error_matrix,'min')

#def spike_times_diff(true_spike_times, estimated_spike_times, min_warping_coef = 1):
#    n_motifs, n_kernels = len(true_spike_times), len(estimated_spike_times)
#    error_matrix = torch.inf*torch.ones([n_kernels,n_motifs])
#    for m in range(n_motifs):
#        for k in range(n_kernels):
#            if min_warping_coef<1:
#                true_t_norm = normalize_times(true_spike_times[m])
#                estimated_t_norm = normalize_times(estimated_spike_times[k])
#                stretch_values = np.linspace(.5, 1.5, 10)
#                diff = np.inf
#                for s in stretch_values:
#                    error = np.sum(np.abs(true_t_norm - estimated_t_norm*s))
#                    if error < diff:
#                        diff = error
#            else:
#                diff = 0
#                for ind in range(len(true_spike_times[m])):
#                    true_neuron, true_time = true_spike_times[m][ind]
#                    estimated_time = estimated_spike_times[k][true_neuron]
#                    diff += abs(true_time-estimated_time)
#            error_matrix[k,m] = diff/len(true_spike_times[m])
#    return find_closest(error_matrix,'min')

def get_similarity(sm, autoencoder, testset_input, metric_names, spike_times=None, verbose=False):

    results = torch.zeros([len(metric_names)])
    results[:] = torch.nan
    learnt_weights = autoencoder.weights.data.clone()
    results = torch.zeros([len(metric_names)])
    results[:] = torch.nan

    if sm.opt['min_warping_coef']<1:
        tensor_spike_times = torch.tensor(sm.spike_times, device=testset_input.device)
        tensor_spike_times[:,:,1] = torch.round((tensor_spike_times[:,:,1]-sm.opt['N_timesteps']//2)*(1+sm.opt['min_warping_coef'])/2+sm.opt['N_timesteps']//2).to(torch.int)
        true_kernels = torch.zeros(sm.SMs.shape, device=testset_input.device)
        true_spike_times = []
        for kernel in range(tensor_spike_times.shape[0]):
            spike_list = []
            for neuron in range(tensor_spike_times.shape[1]):
                spike_list.append((tensor_spike_times[kernel,neuron][0].item(),tensor_spike_times[kernel,neuron][1].item()))
                true_kernels[kernel,tensor_spike_times[kernel,neuron][0].item()] = gaussian_kernel(sm.opt['N_timesteps'],tensor_spike_times[kernel,neuron][1].item(),sm.opt['temporal_jitter'])
            true_spike_times.append(spike_list)
    else:
        true_kernels = sm.SMs
        true_spike_times = sm.spike_times
        
    if len(set(metric_names).intersection(['mean timings similarity','emd mean timings']))>0:
        mean_timings = torch.zeros_like(learnt_weights, device=testset_input.device)
        for ind_motif, spike_times_motif in enumerate(true_spike_times):
            for ind, spike_addresses in enumerate(spike_times_motif):
                mean_timings[ind_motif][spike_addresses] = 1

    for ind_m, metric in enumerate(metric_names):
        if metric == 'kernels similarity':
            results[ind_m] = correlation_kernels(sm.SMs, learnt_weights)
        if metric == 'mean timings similarity':
            results[ind_m] = correlation_mean_timings(mean_timings, learnt_weights,norm=2)
        if metric=='mean time diff':
            if spike_times is not None:
                results[ind_m] = spike_times_diff(true_spike_times, spike_times, min_warping_coef = sm.opt['min_warping_coef'])
            else:
                print('no spike times')
        if metric == 'mse':
            results[ind_m] = kernels_diff(sm.SMs, learnt_weights, 'mse')
        if metric == 'emd':
            results[ind_m] = kernels_diff(sm.SMs, learnt_weights, 'emd')
        if metric == 'emd mean timings':
            results[ind_m] = kernels_diff(mean_timings, learnt_weights, 'emd')
    if verbose:
        for r in range(len(metric_names)):
            print(f'{metric_names[r]} : {np.round(results[r].item(),3)}')
    
    return results