import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch, os
from tqdm import tqdm
from spikeship import spikeship

class SM_World:
    def __init__(self, opt, add_coef = None):
        # initialization of the parameters + drawing of the kernels
        self.opt: Params = opt
        self.kernels = torch.zeros(self.opt.N_kernels, self.opt.N_pre, self.opt.N_delays)
        self.add_coef = add_coef

        for k in range(self.kernels.shape[0]):
            nb_spikes = int(torch.normal(torch.Tensor([self.opt.avg_N_spikes]), torch.Tensor([self.opt.std_N_spikes])).abs())
            addr, delay, precision = (torch.randint(self.opt.N_pre, [nb_spikes]), torch.randint(self.opt.N_delays, [nb_spikes]), torch.normal(torch.ones([nb_spikes])*self.opt.avg_precision, torch.ones([nb_spikes])*self.opt.std_precision).abs())
            for s in range(nb_spikes):
                self.kernels[k, addr[s], :] += gaussian_kernel(self.opt.N_delays, delay[s], 1/precision[s])
            self.kernels[self.kernels>1] = 1

        if add_coef is not None:
            self.draw_mixture(add_coef)
            
    def draw_mixture(self, add_coef):
        add_coef = torch.hstack([add_coef,torch.zeros(1)])
        if self.kernels.shape[0]==self.opt.N_kernels:
            self.kernels = torch.cat((self.kernels, torch.zeros(1, self.kernels.shape[1], self.kernels.shape[2])), 0)
        
        self.kernels[-1, :, :] = torch.matmul(self.kernels.T,add_coef).T
        self.kernels[self.kernels>1] = 1
        self.add_coef = add_coef
    
    def draw_input_one_sm(self, N_trials = 1, mixture_only = False):
        if mixture_only:
            output_labels = self.add_coef.repeat(N_trials,1)
            input_rp = torch.bernoulli(self.kernels[-1,:,:].unsqueeze(0).repeat(N_trials,1,1))
            return input_rp, output_labels
        
        labels = torch.randint(self.kernels.shape[0], [N_trials])
        input_rp = torch.zeros([N_trials, self.opt.N_pre, self.opt.N_timesteps])
        
        if self.add_coef is not None:
            output_labels = self.add_coef.repeat(N_trials,1)
        else:
            output_labels = labels
        for k in range(self.kernels.shape[0]):
            indices = labels == k
            input_rp[indices,:,:] = torch.bernoulli(self.kernels[k,:,:].unsqueeze(0).repeat(indices.sum(),1,1))
            if self.add_coef is not None:
                if k<(self.kernels.shape[0]-1):
                    output_labels[indices,:] = k
                else:
                    output_labels[indices,-1] = k
        return input_rp, output_labels

def tensor2spikeship(rp_tensor):
    N_epochs, N_neurons, N_timesteps = rp_tensor.shape
    #epochs, neurons, times = np.where(rp_tensor>0)
    spike_times = np.array([])
    ii_spike_times = np.zeros([N_epochs, N_neurons, 2])
    nb_previous_timestamps = 0
    last_previous_timestamp = 0
    for e in tqdm(range(N_epochs)):
        neurons, times = np.where(rp_tensor[e,:,:]>0)
        for n in np.unique(neurons):
            indices = np.where(neurons==n)[0]
            ii_spike_times[e,n,:] = [indices[0], indices[-1]+1] + np.ones([2])*nb_previous_timestamps
            spike_times = np.hstack([spike_times,times[indices]]) if spike_times.shape[0]>0 else times[indices]
        nb_previous_timestamps += len(times)
        #last_previous_timestamp += times[-1]
    return spike_times, ii_spike_times

def make_dataset_first_gm(world, N_trials=20, N_coef_changes=5, values=None, normalize_coef=False, mixture_only=False, data_folder = '../Data/'):
    
    path = data_folder+f'synthetic_data_first_gm_{world.opt.N_kernels}_{world.opt.N_delays}_{world.opt.N_pre}_{world.opt.p_input}_{world.opt.avg_precision}_{N_trials}_{N_coef_changes}_{values}_{normalize_coef}_{mixture_only}'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        
    if os.path.exists(path+'_data.npy'):
        print(f'loading files with name {path}*')
        data = np.load(path+'_data.npy')
        labels = np.load(path+'_labels.npy')
        kernels = np.load(path+'_kernels.npy')
        world.kernels = torch.tensor(kernels)
        return torch.tensor(data), torch.tensor(labels), path
        
    #check if saved
    stacked_input = torch.Tensor([])
    stacked_labels = torch.Tensor([])

    if world.add_coef is not None:
        coef_number = world.add_coef.shape[0]-1
    else:
        coef_number = world.kernels.shape[0]
        
    if values is not None:
        ind_val = torch.randint(values.shape[0], [N_coef_changes, coef_number])
        sampled_val = torch.index_select(values, 0, ind_val.flatten()).reshape([N_coef_changes, coef_number])
    else:
        sampled_val = torch.rand([N_coef_changes, coef_number])
    
    if normalize_coef:
        sampled_val = torch.nn.functional.normalize(sampled_val, p=1, dim=1)

    stacked_input, stacked_labels = world.draw_input_one_sm(N_trials = N_trials, mixture_only=mixture_only)

    if world.add_coef is None and N_coef_changes>=1:
        stacked_labels = stacked_labels.unsqueeze(1).repeat(1,coef_number+1)

    for n in range(N_coef_changes):
        world.draw_mixture(sampled_val[n,:])
        input_rp, output_labels = world.draw_input_one_sm(N_trials = N_trials, mixture_only=mixture_only)
        stacked_input = torch.vstack([stacked_input,input_rp])
        stacked_labels = torch.vstack([stacked_labels,output_labels])
        
    np.save(path+'_data', stacked_input.bool())
    np.save(path+'_labels', stacked_labels)
    np.save(path+'_kernels', world.kernels)
    return stacked_input, stacked_labels, path

def make_input_and_plot_sdist(world,
                               N_trials=10,
                               N_coef_changes=200,
                               normalize_coef=True,
                               mixture_only=False,
                               plot=False):
    
    stacked_input, stacked_labels, path = make_dataset_first_gm(world, N_trials=N_trials, N_coef_changes=N_coef_changes, normalize_coef=normalize_coef, mixture_only=mixture_only)
    
    if (stacked_input.sum(dim=(1,2))<1).sum()>0:
        print(f'{(stacked_input.sum(dim=(1,2))<1).sum()} trials with no spike for at least one neuron')
        spike_ind = (stacked_input.sum(dim=(1,2))>=1)
        input_rp, labels = stacked_input[spike_ind,:,:], stacked_labels[spike_ind].numpy()
    else:
        input_rp, labels = stacked_input, stacked_labels.numpy()
    
    if os.path.exists(path+'_spikeship_format.npz'):
        data = np.load(path+'_spikeship_format.npz')
        spike_times, ii_spike_times = data['arr_0'], data['arr_1']
    else:
        spike_times, ii_spike_times = tensor2spikeship(input_rp)
        ii_spike_times = ii_spike_times.astype('int32')
        np.savez(path+'_spikeship_format', spike_times, ii_spike_times)
    
    if os.path.exists(path+'_sdist.npy'):
        S_dist = np.load(path+'_sdist.npy')
    else:
        S_dist = spikeship.distances(spike_times, ii_spike_times)
        np.save(path+'_sdist', S_dist)
    
    if np.isnan(S_dist).sum():
        print(f'{np.isnan(S_dist).sum()/(S_dist.shape[0]**2)*100}% of nan values')
    
    if plot:
        si = np.argsort(labels[:,-1])
        fig, axs = plt.subplots(figsize=(5,5), facecolor='w')
        axs.set_xlabel("Epoch"); axs.set_ylabel("Epoch");
        #axs.set_xticklabels(labels[si][[0,200,400,600,800,1000]]); #axs.set_ylabel("Epoch");
        im = axs.imshow(S_dist[:,si][si], cmap='PuBu')
        axs.set_title("Sorted Dissimilarity Matrix")
        cbar = plt.colorbar(im, ax=axs)
        cbar.set_label("SpikeShip", fontsize=10)
        plt.show();
    return input_rp, labels, S_dist


def plot_raster(raster, trial_nb = 0, colored=False, title = 'raster plot'):

    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    xticks, yticks = 6, 16
    spikelength=.9
    colors = ['grey', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    figsize = (12, 1.61803)
    linewidths=5.0
    
    if colored:
        N_kernels, N_trials, N_neurons, N_timesteps = raster.shape
    else:
        N_trials, N_neurons, N_timesteps = raster.shape

    fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
    if colored:
        for i in range(0, N_neurons):
            for sm in range(N_kernels):
                ax.eventplot(np.where(raster[sm, trial_nb, i, :] > 0)[0],
                    colors=colors[sm], lineoffsets=1.*i+spikelength/2,
                    linelengths=spikelength, linewidths=linewidths)
    else:
        for i in range(0, N_neurons):
            ax.eventplot(np.where(raster[trial_nb, i, :] > 0)[0],
                colors=colors[0], lineoffsets=1.*i+spikelength/2,
                linelengths=spikelength, linewidths=linewidths)

    ax.set_ylabel('address')
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

    ax.grid(visible=True, axis='y', linestyle='-', lw=.5)
    #ax.grid(visible=True, axis='x', which='both', linestyle='-', lw=.1)
    return fig, ax

def plot_SM(SMs, N_show = 5, cmap='plasma', colors=None, aspect=None, figsize = (12, 1.61803)):
    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    N_SMs, N_pre, N_delays = SMs.shape
    steps_pre = N_pre/10

    fig, axs = plt.subplots(1, N_show, figsize=figsize, subplotpars=subplotpars)
    for i_SM in range(N_show):
        ax = axs[i_SM]
        ax.set_axisbelow(True)
        ax.pcolormesh(SMs[i_SM, :, :].flip(1), cmap=cmap, vmin=SMs.min(), vmax=SMs.max())
        #ax.imshow(self.SMs[:, i_SM, :], cmap=cmap, vmin=0, vmax=1, interpolation='none')
        ax.set_xlim(0, N_delays)
        ax.set_xlabel('Delay')
        t = ax.text(.805*N_delays, .95*N_pre, f'#{i_SM+1}', color='k' if colors is None else colors[i_SM])
        t.set_bbox(dict(facecolor='white', edgecolor='white'))
        if not aspect is None: ax.set_aspect(aspect)

        ax.set_ylim(0, N_pre)
        ax.set_yticks(np.arange(0, N_pre, steps_pre)+.5)
        if i_SM>0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(np.arange(0, N_pre, steps_pre)+1)

        for side in ['top', 'right']: ax.spines[side].set_visible(False)
        ax.set_xticks([1, N_delays//3, (N_delays*2)//3])
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(N_delays//4))

    axs[0].set_ylabel('@ Neuron')
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs[:], orientation='vertical', ticks=[0, 1],
                    format=matplotlib.ticker.FixedFormatter(np.round([SMs.min().item(), SMs.max().item()],3)))
    #cbar.set_ticks([0,1]);
    #cbar.ax.set_xticklabels(SMs.min(), SMs.max());
    return fig, axs

def plot_embedding(embedding, labels, title, colors=['r','g','b']):
    fig, ax = plt.subplots(figsize=(10,5))
    for l in np.unique(labels[:,-1]):
        indices = labels[:,-1]==l
        if l<np.unique(labels[:,-1])[-1]:
            ax.scatter(embedding[indices,0], embedding[indices,1], facecolors=colors[int(l)])
        else:
            ax.scatter(embedding[indices,0], embedding[indices,1], facecolors=labels[indices,:3])
    ax.set_title(title);

def plot_3d_embedding(embedding, labels, title, colors=['r','g','b'], view_init=[0,90]):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    for l in np.unique(labels[:,-1]):
        indices = labels[:,-1]==l
        if l<np.unique(labels[:,-1])[-1]:
            ax.scatter(embedding[indices,0], embedding[indices,1], embedding[indices,2], facecolors=colors[int(l)])
        else:
            ax.scatter(embedding[indices,0], embedding[indices,1], embedding[indices,2], facecolors=labels[indices,:3])
    ax.set_title(title);
    ax.view_init(view_init[0], view_init[1])

def plot_corrcoef(world, estimated_sms, cmap='viridis'):
    fig, ax = plt.subplots(1,1)#, figsize=(5,5))
    corr_matrix = torch.corrcoef(torch.vstack([world.kernels[:-1,:,:].flatten(start_dim=1), torch.tensor(estimated_sms).flatten(start_dim=1)]))
    #print(corr_matrix.min(), corr_matrix.max())
    ax.imshow(corr_matrix, cmap=cmap);
    ax.set_xticks(np.arange(world.kernels.shape[0]-1+estimated_sms.shape[0],1))
    ax.set_yticks(np.arange(world.kernels.shape[0]-1+estimated_sms.shape[0]), labels=[f'true \#{n}' for n in range(world.kernels.shape[0]-1)]+[f'comp \#{n+1}' for n in range(world.kernels.shape[0]-1)])
    #ax.set_yticklabels([f'true #{n}' for n in range(world.kernels.shape[0]-1)]+[f'comp #{n+1}' for n in range(world.kernels.shape[0]-1)])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, orientation='vertical', ticks=[0,1], format=matplotlib.ticker.FixedFormatter(np.round([corr_matrix.min().item(), corr_matrix.max().item()],3)))


def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))