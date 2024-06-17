import torch, os, matplotlib
import matplotlib.pyplot as plt
import numpy as np

class default_parameters():
    seed = 666
    
    N_pre = 10 # number of neurons
    N_timesteps = 1000 # number of timesteps for the raster plot (in ms)

    N_delays = 51 # number of timesteps in spiking motifs, must be a odd number for convolutions
    N_SMs = 5 # number of structured spiking motifs
    N_involved = 5*torch.ones(N_SMs) # number of neurons involved in the spiking motif
    avg_fr = 20 # average firing rate of the neurons (in Hz)
    std_fr = 1 # standard deviation for the firing rates of the different neurons
    frs = torch.normal(avg_fr, std_fr, size=(N_pre,)).abs()
    freq_sms = 2*torch.ones(N_SMs) # frequency of apparition of the different spiking motifs (in Hz)
    overlapping_sms = True # possibility to have overlapping sequences

    temporal_jitter = .1 # temporal jitter for the spike generation in motifs
    dropout_proba = .5 # probabilistic participations of the different neurons to the spiking motif
    additive_noise = 0 # percentage of background noise/spontaneous activity
    warping_coef = 1 # coefficient for time warping

    def get_parameters(self):
        return f'{self.N_pre}_{self.N_delays}_{self.N_SMs}_{self.N_timesteps}_{self.N_involved.mean()}_{self.avg_fr}_{self.freq_sms.mean()}_{self.overlapping_sms}_{self.temporal_jitter}_{self.dropout_proba}_{self.additive_noise}_{self.warping_coef}_{self.seed}'

def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))
    
class sm_generative_model:
    def __init__(self, opt, device='cpu'):
        
        self.opt = opt
        self.device = device
        torch.manual_seed(opt.seed)
        # initialization of the different spiking motifs probability distributions
        self.SMs = torch.zeros(self.opt.N_SMs, self.opt.N_pre, self.opt.N_delays, device = device)
        # average probability of having a spike for a specific timestep (homogeneous Poisson)
        self.opt.frs = self.opt.frs.to(device)
        proba_timestep = self.opt.frs*1e-3
        # compute the average number of spike per motif for each neuron
        N_spikes_per_motif = proba_timestep*self.opt.N_delays
        
        for k in range(self.opt.N_SMs):
            # get the indices of neurons participating in the motif
            if self.opt.N_involved[k]<self.opt.N_pre:
                all_ind = torch.randperm(self.opt.N_pre)
                self.ind_inv = all_ind[:int(self.opt.N_involved[k])]
                ind_not_inv = all_ind[int(self.opt.N_involved[k]):]
                self.SMs[k,ind_not_inv] = 1
            else:
                self.ind_inv = torch.arange(self.opt.N_pre)
            
            # draw probability distributions with gaussian kernels
            for n in self.ind_inv:
                spike_times = torch.randint(self.opt.N_delays, [int(torch.round(N_spikes_per_motif[n]))])
                for s in range(len(spike_times)):
                    self.SMs[k, n] += gaussian_kernel(self.opt.N_delays, spike_times[s], self.opt.temporal_jitter).to(device)
        
        # normalize kernels to have each row suming to 1 (probability distribution)
        self.SMs.div_(torch.norm(self.SMs, p=1, dim=-1, keepdim=True)+1e-14)

        # adding background activity
        self.SMs = (1-self.opt.additive_noise)*self.SMs+self.opt.additive_noise*torch.ones_like(self.SMs)/self.opt.N_delays
    
    def draw_input(self, nb_trials=10):
        
        # initialize input and output tensors
        input_proba = -1*torch.ones(nb_trials, self.opt.N_pre, self.opt.N_timesteps, device=self.device)
        output_rp = torch.zeros(nb_trials, self.opt.N_SMs, self.opt.N_timesteps, device=self.device)
        
        nb_motifs = torch.round(nb_trials*self.opt.N_timesteps*self.opt.freq_sms*1e-3)
        if self.opt.overlapping_sms:
        # iterate over the different occurence of motifs to modify the local distribution
            for k in range(self.opt.N_SMs):
                nb_motif_k = int(nb_motifs[k])
                for n in range(nb_motif_k):
                    time = torch.randint(self.opt.N_delays//2, self.opt.N_timesteps-(self.opt.N_delays//2+1), [1])
                    trial = torch.randint(nb_trials, [1])
                    
                    previous = input_proba[trial,:,time-(self.opt.N_delays//2):time+self.opt.N_delays//2+1].squeeze(0)
                    new = torch.zeros_like(previous, device=self.device)
                    # get dropped indices
                    if self.opt.dropout_proba:
                        not_to_keep = torch.where(torch.bernoulli(torch.ones_like(self.ind_inv)*self.opt.dropout_proba)==1)[0]
                        not_kept_ind = self.ind_inv[not_to_keep]
                        motif = self.SMs[k]
                        motif[not_kept_ind] = 1/self.opt.N_delays
                    else:
                        motif = self.SMs[k]
                    # if no prior distribution on the location of the motif replace by motif distribution
                    new[previous==-1]=motif[previous==-1]
                    # otherwise formula to merge probabilities of independent Bernoulli process:
                    #    Bernoulli(p) + Bernoulli(q) = Bernoulli(p+q-p*q)
                    new[previous!=-1]=motif[previous!=-1]+previous[previous!=-1]-previous[previous!=-1]*motif[previous!=-1]

                    input_proba[trial,:,time-(self.opt.N_delays//2):time+self.opt.N_delays//2+1] = new
                    output_rp[trial,k,time] = 1

        else:
        # generate list of (trial,timelocation) to draw the motifs
            loc_grid = torch.ones(nb_trials,self.opt.N_timesteps//self.opt.N_delays)
            nb_added = 0
            nb_motifs_distrib = nb_motifs/nb_motifs.sum()
            while nb_added<nb_motifs.sum() and loc_grid.sum()>0:
                # get motif type
                k = nb_motifs_distrib.multinomial(num_samples=1)
                # possible locations to avoid overlapping
                poss_loc_trial, poss_loc_time = torch.where(loc_grid==1)
                ind_loc = torch.randint(len(poss_loc_trial), [1])
                trial, time = poss_loc_trial[ind_loc], self.opt.N_delays*poss_loc_time[ind_loc]+self.opt.N_delays//2
                if self.opt.dropout_proba:
                    not_to_keep = torch.where(torch.bernoulli(torch.ones_like(self.ind_inv)*self.opt.dropout_proba)==1)[0]
                    not_kept_ind = self.ind_inv[not_to_keep]
                    motif = self.SMs[k].squeeze(0)
                    motif[not_kept_ind] = 1/self.opt.N_delays
                else:
                    motif = self.SMs[k]
                input_proba[trial,:,time-(self.opt.N_delays//2):time+self.opt.N_delays//2+1] = motif
                output_rp[trial,k,time] = 1
                nb_added += 1
                loc_grid[trial,poss_loc_time[ind_loc]] = 0

        # random distribution when no modification was added
        input_proba[input_proba==-1] = 1/self.opt.N_delays
        # normalizing to required firing rates.cpu().numpy()
        input_proba = input_proba.div_(torch.norm(input_proba, p=1, dim=-1, keepdim=True))*(self.opt.frs.unsqueeze(0).unsqueeze(-1).repeat(nb_trials,1,self.opt.N_timesteps)*1e-3*self.opt.N_timesteps)
        # harsh threshold to have a single spike in one timebin
        input_proba[input_proba>1] = 1
        # Bernoulli trial on the probability distribution
        input_rp = torch.bernoulli(input_proba)
                    
        return input_rp, output_rp


def generate_dataset(parameters,num_samples,record_path='../synthetic_data/', verbose=True,  device='cpu'):
    
    if not os.path.exists(record_path):
        os.mkdir(record_path)
        
    # create into train (80%) and test (20%) sets
    num_train, num_test = int(.8*num_samples), int(.2*num_samples)
    model_path = record_path+f'generative_model_'+parameters.get_parameters()
    trainset_path = record_path+f'synthetic_rp_trainset_{num_train}_'+parameters.get_parameters()+'.pt'
    testset_path = record_path+f'synthetic_rp_testset_{num_test}_'+parameters.get_parameters()+'.pt'
    if os.path.exists(model_path):
        sm = torch.load(model_path, map_location = device)
        sm.device = device
    else:
        sm = sm_generative_model(parameters, device=device)
        torch.save(sm, model_path)

    for dataset in ['trainset', 'testset']:
        if dataset=='trainset':
            num = num_train
        elif dataset=='testset':
            num = num_test
        dataset_path = record_path+f'synthetic_rp_{dataset}_{num}_'+parameters.get_parameters()+'.pt'
        if verbose: print(dataset_path)
        if os.path.exists(dataset_path):
            dataset_input_list, dataset_output_list = torch.load(dataset_path, map_location=device)
            dataset_input = torch.zeros(num,parameters.N_pre,parameters.N_timesteps, device=device)
            dataset_output = torch.zeros(num,parameters.N_SMs,parameters.N_timesteps, device=device)
            dataset_input[dataset_input_list] = 1
            dataset_output[dataset_output_list] = 1
        else:
            dataset_input, dataset_output = sm.draw_input(nb_trials=num)
            dataset_input_list = torch.where(dataset_input==1)
            dataset_output_list = torch.where(dataset_output==1)
            torch.save((dataset_input_list, dataset_output_list), dataset_path)
        if dataset=='trainset':
            trainset_input, trainset_output = dataset_input, dataset_output
        elif dataset=='testset':
            testset_input, testset_output = dataset_input, dataset_output
    return sm, trainset_input, trainset_output, testset_input, testset_output


def plot_SM(SMs, N_show = 5, order_sms= False, cmap='Purples', colors=None, aspect=None, figsize = (12, 1.61803)):
    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    SMs = SMs.to('cpu')

    cmap_2 = matplotlib.colormaps['Set3']
    
    N_SMs, N_pre, N_delays = SMs.shape
    steps_pre = N_pre/10
    N_show = min(N_show, N_SMs)
        
    fig, axs = plt.subplots(1, N_show, figsize=figsize, subplotpars=subplotpars)
    for i_SM in range(N_show):
        if N_show>1:
            ax = axs[i_SM]
        else:
            ax = axs
        ax.set_axisbelow(True)
        if order_sms:
            ordered_sm = SMs[i_SM,SMs[i_SM].argmax(dim=1).argsort(),:]
            ax.pcolormesh(ordered_sm, cmap=cmap, vmin=SMs.min(), vmax=SMs.max())
        else:
            ax.pcolormesh(SMs[i_SM], cmap=cmap, vmin=SMs.min(), vmax=SMs.max())
        ax.set_xlim(0, N_delays)
        ax.set_xlabel('Delay')
        t = ax.text(.1*N_delays, .95*N_pre, f'#{i_SM+1}', color='k' if colors is None else colors[i_SM])
        t.set_bbox(dict(facecolor=cmap_2(i_SM), edgecolor='black'))
        if not aspect is None: ax.set_aspect(aspect)

        ax.set_ylim(0, N_pre)
        ax.set_yticks(np.arange(0, N_pre, 1)+.5)
        ax.set_yticklabels('')

        for side in ['top', 'right']: ax.spines[side].set_visible(False)
        ax.set_xticks([1, N_delays//3, (N_delays*2)//3])
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(N_delays//4))

    if N_show>1:
        axs[0].set_ylabel('@ Neuron')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs[:], orientation='vertical', ticks=[0, 1],
                format=matplotlib.ticker.FixedFormatter(np.round([SMs.min().item(), SMs.max().item()],3)))
    else:
        axs.set_ylabel('@ Neuron')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs, orientation='vertical', ticks=[0, 1],
                format=matplotlib.ticker.FixedFormatter(np.round([SMs.min().item(), SMs.max().item()],3)))

    return fig, axs

def plot_raster(raster, title = 'raster plot'):

    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    xticks, yticks = 6, 16 
    spikelength=.9
    colors = ['black']
    figsize = (12, 1.61803)
    linewidths=1.0
    
    N_neurons, N_timesteps = raster.shape

    raster = raster.to('cpu')

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
    for i in range(0, N_neurons):
        ax.eventplot(np.where(raster[i, :] == 1.)[0], 
            colors=colors, lineoffsets=1.*i+spikelength/2, 
            linelengths=spikelength, linewidths=linewidths)

    ax.set_ylabel('@ Neuron')
    ax.set_xlabel('Time (a. u.)')
    ax.set_xlim(0, N_timesteps)
    ax.set_ylim(0, N_neurons)

    ax.set_yticks(np.arange(0, N_neurons, 1)+.5)
    ax.set_yticklabels('')#np.linspace(1, N_neurons, 9, endpoint=True).astype(int))
    for side in ['top', 'right']: ax.spines[side].set_visible(False)

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(N_timesteps/4))
    ax.set_xticks(np.linspace(1, N_timesteps, xticks, endpoint=True))
    ax.set_xticklabels(np.linspace(1, N_timesteps, xticks, endpoint=True).astype(int))
    ax.set_title(title)
    
    #ax.grid(visible=True, axis='y', linestyle='-', lw=.5)
    #ax.grid(visible=True, axis='x', which='both', linestyle='-', lw=.1)

    return fig, ax

def plot_colored_raster(input_rp, output_rp, N_delays, indice_trial = 0, title = 'raster plot'):
    cmap_2 = matplotlib.colormaps['Set3']
    fig, ax = plot_raster(input_rp[indice_trial], title = title)
    indices = np.where(output_rp[indice_trial]==1)
    for k in range(len(indices[0])):
        ax.axvspan(indices[1][k]-N_delays//2, indices[1][k]+N_delays//2+1, facecolor=cmap_2(indices[0][k]), alpha=0.5)
    return fig, ax

def plot_results(world, loss, trained_layer_of_neurons, input_raster_plot, true_occurence, moving_window = 50, order_sms = False, plot = True, verbose = True, device = 'cpu'):

    factors, reconstruction = trained_layer_of_neurons(input_raster_plot)
    similarity_factors, similarity_kernels, mse, emd = get_similarity(world, trained_layer_of_neurons, input_raster_plot, device=device)
    if verbose:
        print(f'======    Results    ====== \nFactors similarity : {np.round(similarity_factors,4)}\nKernels similarity : {np.round(similarity_kernels,4)}\nMSE                : {np.round(mse,4)}\nEMD                : {np.round(emd,4)}\n')

    if plot:
        cmap = matplotlib.colormaps['Set3']
        cmap_rec = matplotlib.colormaps['Purples']
        
        nb_motifs = int(true_occurence[0].sum())
        N_SMs, N_pre, N_delays = trained_layer_of_neurons.kernel_size
        if torch.is_tensor(N_SMs): N_SMs = N_SMs.item()
        fig_loss, ax_loss = plt.subplots(figsize=(12.75,2));
        ax_loss.plot(moving_average(loss,moving_window));
        ax_loss.set_title('loss');
        figsize_kernels = (12*N_SMs/world.SMs.shape[0],2)
        weights = trained_layer_of_neurons.decoding_weights.data
        plot_SM(weights.squeeze(1).detach().cpu(), figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);
        plot_SM(world.SMs.detach().cpu(), figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);
    
        N_sample, _, _ = input_raster_plot.shape
        random_ind = torch.randint(N_sample,[1])
        fig_raster, ax_raster = plot_colored_raster(input_raster_plot[random_ind], true_occurence[random_ind].cpu().numpy(), world.opt.N_delays);
        padded_factors = torch.nn.functional.pad(factors[random_ind], (N_delays//2,N_delays//2, 0, 0), mode='constant')
        for k in range(N_SMs):
            ax_raster.plot((N_pre/padded_factors.max().item())*padded_factors[0,k].detach().cpu().numpy())
    
    return similarity_factors, similarity_kernels, mse, emd

    