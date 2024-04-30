import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))

def homeo_gain_control(histo, lambda_homeo=.25):
    gain = torch.exp(lambda_homeo*(1-len(histo)*histo))
    return gain

def topk_p_raster(p_out, k):
    values, indices = torch.topk(p_out.flatten(), k)
    spike_indices = np.array(np.unravel_index(indices.cpu().numpy(), p_out.shape)).T
    output_spikes = torch.zeros_like(p_out)
    output_spikes[spike_indices[:,0], spike_indices[:,1]] = 1
    return output_spikes
    
class SM_World:
    def __init__(self, opt):
        self.opt: Params = opt
        self.SMs = torch.zeros(self.opt.N_SMs, self.opt.N_pre, self.opt.N_delays)
        for k in range(self.opt.N_SMs):
            nb_spikes = int(torch.normal(torch.Tensor([self.opt.avg_N_spikes]), torch.Tensor([self.opt.std_N_spikes])))
            addr, delay, precision = (torch.randint(self.opt.N_pre, [nb_spikes]), torch.randint(self.opt.N_delays, [nb_spikes]), torch.normal(torch.ones([nb_spikes])*self.opt.avg_precision, torch.ones([nb_spikes])*self.opt.std_precision).abs())
            for s in range(nb_spikes):
                self.SMs[k, addr[s], :] += gaussian_kernel(self.opt.N_delays, delay[s], 1/precision[s])
            if self.SMs[k, :, :].max()>1: self.SMs[k, :,:]/=self.SMs[k, :, :].max()
                
    def draw_input(self):
        A = torch.zeros(self.opt.N_pre, self.opt.N_timesteps)
        B = torch.zeros(self.opt.N_SMs, self.opt.N_timesteps)
        
        a_noise, time_noise =  torch.randint(self.opt.N_pre, [int(self.opt.avg_N_noise)]), torch.randint(self.opt.N_timesteps, [int(self.opt.avg_N_noise)])
        A[a_noise, time_noise] = 1
        
        nb_motifs = int(torch.normal(torch.Tensor([self.opt.avg_nb_motifs]), torch.Tensor([self.opt.std_nb_motifs])))
        for i in range(nb_motifs):
            k_sm, time_sm =  torch.randint(self.opt.N_SMs, [1]), torch.randint(self.opt.N_delays, self.opt.N_timesteps, [1]) # check that there is no overlapping check for outliers
            if time_sm<self.opt.N_delays:
                A[:,:time_sm] = torch.bernoulli(self.SMs[k_sm, :, :time_sm].squeeze(0))*(k_sm+2)
            else:
                A[:,time_sm-self.opt.N_delays:time_sm] = torch.bernoulli(self.SMs[k_sm, :, :].squeeze(0))*(k_sm+2)
            B[k_sm, time_sm] = 1
        return A, B
    
    
class HDSNN1D(torch.nn.Module):
    def __init__(self, N_pre, N_delays, N_neurons, SMs=None, do_bias=True, output='sigma', homeo_gain=False, threshold=None, tau = 10, device='cpu'): 
        super(HDSNN1D, self).__init__()
        self.device=device
        self.kernel_size = (N_neurons, N_pre, N_delays)
        padding_conv = N_delays
        padding_transposed = 0
        padding_mode = 'zeros'
        self.threshold = threshold
        self.homeo_gain = homeo_gain
        self.cumhisto = (torch.ones([N_neurons,1])/N_neurons).to(device)
        self.tau = tau

        self.conv_layer = torch.nn.Conv1d(N_pre, N_neurons, kernel_size=self.kernel_size, padding=padding_conv, padding_mode=padding_mode, bias=do_bias, device=device)
        
        if output=='sigma': 
            self.nl = torch.nn.Sigmoid()
        else: 
            self.nl = torch.nn.Softmax(dim=0)
            
        self.reconstruction_layer = torch.nn.ConvTranspose1d(N_neurons, N_pre, kernel_size=self.kernel_size, padding=padding_transposed, bias=do_bias, device=device)
        
        if SMs is not None:
            weight_init = SMs
            self.conv_layer.weight = self.reconstruction_layer.weight = torch.nn.Parameter(weight_init.to(device), requires_grad=True)
        else:
            self.conv_layer.weight = self.reconstruction_layer.weight = torch.nn.Parameter(torch.rand(self.kernel_size).to(device), requires_grad=True)
        
    def forward(self, a, k): 
        # convolution with LR
        logit_b = self.conv_layer(a)[:,:-self.kernel_size[2]-1]
        p_b = self.nl(logit_b)
        if self.homeo_gain:
            p_b = homeo_gain_control(self.cumhisto).to(self.device)*p_b
        
        # select the top k values for spikes
        if self.threshold:
            output_spikes = p_b>self.threshold
        else: 
            output_spikes = topk_p_raster(p_b, k)
        
        # recontruct the input with logits
        estimated_input = self.reconstruction_layer(output_spikes.to(torch.float32))
        estimated_input = torch.roll(estimated_input, -self.kernel_size[2], dims=1)[:,:-self.kernel_size[2]+1]
        
        if (self.conv_layer.weight-self.reconstruction_layer.weight).sum()!=0:
            print('ERROR: weights are not shared')
            
        if self.homeo_gain:
            self.cumhisto[:,0] = (1-1/self.tau)*self.cumhisto[:,0]+1/self.tau*output_spikes.sum(dim=1)/output_spikes.sum()
        
        return p_b, output_spikes, estimated_input
    
    
def plot_SM(SMs, N_show = 5, cmap='plasma', colors=None, aspect=None, figsize = (12, 1.61803)):
        
        subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

        N_SMs, N_pre, N_delays = SMs.shape
        
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
            ax.set_yticks(np.arange(0, N_pre, 1)+.5)
            if i_SM>0: 
                ax.set_yticklabels([])
            else:
                ax.set_yticklabels(np.arange(0, N_pre, 1)+1)

            for side in ['top', 'right']: ax.spines[side].set_visible(False)
            ax.set_xticks([1, N_delays//3, (N_delays*2)//3])
            ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(N_delays//4))

        axs[0].set_ylabel('@ Neuron')
        return fig, axs
    
def plot_raster(raster, colored=False, title = 'raster plot'):

    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    xticks, yticks = 6, 16 
    spikelength=.9
    colors = ['grey', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
 '#7f7f7f', '#bcbd22', '#17becf']
    figsize = (12, 1.61803) 
    linewidths=1.0
    
    N_neurons, N_timesteps = raster.shape
    label_max = int(raster.unique()[-1].item())+1
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplotpars=subplotpars)
    if colored:
        for i in range(0, N_neurons):
            for sm in range(1,label_max):
                ax.eventplot(np.where(raster[i, :] == sm)[0], 
                    colors=colors[sm-1], lineoffsets=1.*i+spikelength/2,
                    linelengths=spikelength, linewidths=linewidths)
    else:
        for i in range(0, N_neurons):
            ax.eventplot(np.where(raster[i, :] > 0)[0], 
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

def plot_results(world, loss, trained_layer_of_neurons):
    plot_SM(world.SMs, figsize = (12, 2));
    nb_motifs = int(torch.normal(torch.Tensor([world.opt.avg_nb_motifs]), torch.Tensor([world.opt.std_nb_motifs])))
    a, true_b = world.draw_input(nb_motifs)

    fig_loss, ax_loss = plt.subplots(figsize=(12.75,2));
    ax_loss.plot(loss);
    ax_loss.set_title('loss');
    plot_SM(trained_layer_of_neurons.conv_layer.weight.data.squeeze(1).detach().cpu().flip(2), figsize=(12,2), N_show = trained_layer_of_neurons.kernel_size[0]);
    b, spikes, estimated_input = trained_layer_of_neurons(a, nb_motifs)
    plot_raster(a.detach().cpu().numpy(), title = 'input raster plot');
    plot_raster(spikes.detach().cpu().numpy(), title = 'detection of spiking motifs');
    fig, ax = plt.subplots(2, 1, figsize=(12.6,4))
    for i in range(len(b)):
        ax[0].plot(b[i].detach().cpu().numpy());
    ax[1].imshow(estimated_input.detach().cpu().numpy(), aspect='auto');

    ax[0].set_xticks([]);
    ax[0].set_xlim(0,b.shape[1]);
    ax[0].set_title('output probability');
    ax[1].set_title('reconstructed input');
    plt.show();