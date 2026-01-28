import torch
import numpy as np
from wassa_utils import find_closest, correlation_latent_variables, correlation_kernels

def torch_cdf_loss(reconstructed,input_seq,zeros,normalize):
    ''' Computes the 1D-Wasserstein distance along the last dimension of reconstructed and input_seq
        input_seq and reconstructed : BxNxT tensor where B is the number of batches, N the number of 
                                      neurons (or channels of the input) and T the number of timesteps
        zeros : treat neurons with no spike as informative ('same') or not ('ignore') -> nan
    '''
    assert reconstructed.shape == input_seq.shape
    
    cdf_reconstructed = torch.cumsum(reconstructed,dim=-1)
    cdf_input = torch.cumsum(input_seq,dim=-1)

    if normalize:
        norm_rec = cdf_reconstructed[:,:,-1].clone().unsqueeze(-1).repeat(1,1,reconstructed.shape[-1])
        norm_seq = cdf_input[:,:,-1].clone().unsqueeze(-1).repeat(1,1,input_seq.shape[-1])
        if zeros == 'same':
            norm_rec[norm_rec==0], norm_seq[norm_seq==0] = 1, 1
        elif zeros == 'ignore':
            norm_rec[norm_rec==0], norm_seq[norm_seq==0] = torch.nan, torch.nan
        cdf_input.div_(norm_seq), cdf_reconstructed.div_(norm_rec)
    elif zeros == 'ignore':
        norm_rec[norm_rec.sum(dim=-1)==0], norm_seq[norm_seq.sum(dim=-1)==0] = torch.nan, torch.nan
    cdf_distance = torch.mean(torch.abs(cdf_input-cdf_reconstructed),dim=-1)
    return cdf_distance

class WassDist(torch.nn.Module):
    def __init__(self,zeros='same',normalize=True):
        super().__init__()
        self.zeros = zeros
        self.normalize = normalize

    def forward(self, input_seq, target):
        return torch_cdf_loss(target,input_seq,self.zeros,self.normalize).nanmean()

def kernels_diff(true_kernels, learnt_kernels, metric):
    
    n_motifs, n_neurons, n_timbin = true_kernels.shape
    n_kernels = learnt_kernels.shape[0]

    if metric == 'mse':
        true_matrix = true_kernels.flatten(start_dim=1).unsqueeze(0).repeat(n_kernels,1,1)
        learnt_matrix = learnt_kernels.flatten(start_dim=1).unsqueeze(1).repeat(1,n_motifs,1)
        error_matrix = ((true_matrix-learnt_matrix)**2).mean(axis=-1)
    elif metric == 'emd':
        true_kernels.div_(torch.norm(true_kernels, p=1, dim=(2), keepdim=True)+1e-14)
        learnt_kernels.div_(torch.norm(learnt_kernels, p=1, dim=(2), keepdim=True)+1e-14)
        true_matrix = true_kernels.unsqueeze(0).repeat(n_kernels,1,1,1)
        learnt_matrix = learnt_kernels.unsqueeze(1).repeat(1,n_motifs,1,1)
        error_matrix = torch_cdf_loss(true_matrix,learnt_matrix).mean(axis=-1)

    return find_closest(error_matrix,'min')

def get_similarity(sm, autoencoder, testset_input, metric_names, verbose=False):

    results = torch.zeros([len(metric_names)])
    results[:] = torch.nan
    if autoencoder.tied_weights:
        learnt_weights = autoencoder.weights.data.clone()
    else:
        learnt_weights = autoencoder.decoder[0].weight.data.clone()

    if len(set(metric_names).intersection(['mean timings similarity','emd mean timings']))>0:
        mean_timings = torch.zeros(sm.SMs.shape, device=testset_input.device)
        for ind, spike_address in enumerate(sm.spike_times):
            mean_timings[spike_address] = 1

    for ind_m, metric in enumerate(metric_names):
        if metric == 'factors similarity':
            true_factors = torch.nn.functional.conv1d(testset_input,sm.SMs)
            learnt_factors = torch.nn.functional.conv1d(testset_input,learnt_weights)
            results[ind_m] = correlation_latent_variables(true_factors, learnt_factors)
        if metric == 'kernels similarity':
            results[ind_m] = correlation_kernels(sm.SMs, learnt_weights)
        if metric == 'mean timings similarity':
            results[ind_m] = correlation_kernels(mean_timings, learnt_weights, p='max')/mean_timings.sum()
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