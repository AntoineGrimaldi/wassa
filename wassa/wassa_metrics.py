import torch
import numpy as np
from wassa.wassa import WassA

def torch_cdf_loss(reconstructed,input_seq,zeros='same',p=1):

    assert reconstructed.shape == input_seq.shape
    
    cdf_reconstructed = torch.cumsum(reconstructed,dim=-1)
    cdf_input = torch.cumsum(input_seq,dim=-1)

    if zeros=='ignore':
        ind_nonzero_spikes = (torch.sum(reconstructed, dim=-1)>0) * (torch.sum(input_seq, dim=-1)>0)
        diff_cdf = (cdf_reconstructed-cdf_input)[ind_nonzero_spikes]
    elif zeros=='same':
        diff_cdf = cdf_reconstructed-cdf_input

    if p == 1:
        cdf_distance = torch.sum(torch.abs(diff_cdf),dim=-1)
    else:
        # not the real Wasserstein distance but a powered EMD
        cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(diff_cdf),p),dim=-1),1/p)
    return cdf_distance

class WassDist(torch.nn.Module):
    def __init__(self, p=1, zeros='ignore'):
        super(WassDist, self).__init__()
        self.p = p
        self.zeros = zeros

    def forward(self, input_seq, target):
        N_batch, N_neuron, N_timestep = target.shape
        return torch_cdf_loss(input_seq,target,zeros=self.zeros,p=self.p).sum()/(N_batch*N_neuron*N_timestep)
    
def find_closest(matrix,max_or_min):
    # size of the matrix is NxM where N is the number of trained kernels and M is the number of ground truth motifs
    # loop over the different motifs to see if similarity is good with different kernels 
    #      - takes the best score for one motifs and discard the possibilities to match with other kernels
    #      - accumulate similarity value
    value = 0
    for l_k in range(matrix.shape[0]):
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
    return value

def correlation_latent_variables(true, learnt):

    n_batch, n_kernels, n_timesteps = learnt.shape
    _, n_motifs, _ = true.shape

    shaped_true = true.swapaxes(0,1).flatten(start_dim=1).unsqueeze(0).repeat(n_kernels,1,1)
    shaped_learnt = learnt.swapaxes(0,1).flatten(start_dim=1).unsqueeze(1).repeat(1,n_motifs,1)

    mean_true = shaped_true.mean(axis=-1).unsqueeze(-1).repeat(1,1,n_batch*n_timesteps)
    mean_learnt = shaped_learnt.mean(axis=-1).unsqueeze(-1).repeat(1,1,n_batch*n_timesteps)

    var_true = ((shaped_true-mean_true)**2).mean(axis=-1)
    var_learnt = ((shaped_learnt-mean_learnt)**2).mean(axis=-1)
    covar = ((shaped_learnt-mean_learnt)*(shaped_true-mean_true)).mean(axis=-1)

    corrcoef = covar/(torch.sqrt(var_true*var_learnt)+1e-14)
    max_correlation = find_closest(corrcoef,'max')

    return max_correlation/n_kernels

def correlation_kernels(true, learnt, norm='2'):

    if norm=='2':
        true = true.div_(torch.norm(true, p=2, dim=(1,2), keepdim=True)+1e-14)
        learnt = learnt.div_(torch.norm(learnt, p=2, dim=(1,2), keepdim=True)+1e-14)
    elif norm=='1':
        true = true.div_(torch.norm(true, p=1, dim=(1,2), keepdim=True)+1e-14)
        learnt = learnt.div_(torch.norm(learnt, p=1, dim=(1,2), keepdim=True)+1e-14)
    elif norm=='max':
        true = true.div_(torch.amax(true, dim=(1,2), keepdim=True))
        learnt = learnt.div_(torch.amax(learnt, dim=(1,2), keepdim=True))
    n_kernels, n_neurons, n_delays = learnt.shape
    n_motifs, _, _ = true.shape

    padded_true = torch.nn.functional.pad(true, (torch.div(n_delays,2,rounding_mode='floor'),torch.div(n_delays,2,rounding_mode='floor'),0,0,0,0), mode='constant')
    cross_correlation = torch.nn.functional.conv1d(padded_true,learnt)
    max_cross_correlation = cross_correlation.amax(dim=-1).T

    max_correlation = find_closest(max_cross_correlation,'max')
    
    if norm=='max':
        max_correlation /= n_neurons

    return max_correlation/n_kernels

def kernels_diff(true_kernels, learnt_kernels, metric):
    
    n_motifs, n_neurons, n_timbin = true_kernels.shape
    n_kernels = learnt_kernels.shape[0]

    if metric == 'mse':
        true_matrix = true_kernels.flatten(start_dim=1).unsqueeze(0).repeat(n_kernels,1,1)
        learnt_matrix = learnt_kernels.flatten(start_dim=1).unsqueeze(1).repeat(1,n_motifs,1)
        error_matrix = ((true_matrix-learnt_matrix)**2).sum(axis=-1)/(n_timbin*n_neurons)
    elif metric == 'emd':
        true_kernels.div_(torch.norm(true_kernels, p=1, dim=(2), keepdim=True)+1e-14)
        learnt_kernels.div_(torch.norm(learnt_kernels, p=1, dim=(2), keepdim=True)+1e-14)
        true_matrix = true_kernels.unsqueeze(0).repeat(n_kernels,1,1,1)
        learnt_matrix = learnt_kernels.unsqueeze(1).repeat(1,n_motifs,1,1)
        error_matrix = torch_cdf_loss(true_matrix,learnt_matrix).sum(axis=-1)/(n_neurons*n_timbin)

    min_diff = find_closest(error_matrix,'min')
    
    return min_diff/n_kernels

def get_similarity(sm, autoencoder, testset_input, device='cpu'):
    
    generative_model_ae = WassA((sm.opt.N_SMs, sm.opt.N_pre, sm.opt.N_delays), sm.SMs, device=device)
    true_factors, _ = generative_model_ae(testset_input)
    learnt_factors, _ = autoencoder(testset_input)
    learnt_weights = autoencoder.decoding_weights.detach().clone()
    
    similarity_factors = correlation_latent_variables(true_factors.detach(), learnt_factors.detach())
    similarity_kernels = correlation_kernels(sm.SMs, learnt_weights)
    mse = kernels_diff(sm.SMs, learnt_weights, 'mse')
    emd = kernels_diff(sm.SMs, learnt_weights, 'emd')
    mean_timings = torch.zeros(sm.SMs.shape, device=device)
    for ind, spike_address in enumerate(sm.spike_times):
        mean_timings[spike_address] = 1
    similarity_mean_timings = correlation_kernels(mean_timings, learnt_weights, norm='max')
    
    emd_mean_timings = kernels_diff(mean_timings, learnt_weights, 'emd')
    
    return similarity_factors, similarity_kernels, similarity_mean_timings, mse, emd, emd_mean_timings

