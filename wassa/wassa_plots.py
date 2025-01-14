import matplotlib, torch
import matplotlib.pyplot as plt
import numpy as np
from wassa.wassa import WassA
from wassa.wassa_metrics import get_similarity

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

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
    similarity_factors, similarity_kernels, similarity_means, mse, emd, emd_means = get_similarity(world, trained_layer_of_neurons, input_raster_plot, device=device)
    if verbose:
        print(f'======    Results    ====== \nFactors similarity : {np.round(similarity_factors.cpu().numpy(),4)}\nKernels similarity : {np.round(similarity_kernels.cpu().numpy(),4)}\nMean similarity : {np.round(similarity_means.cpu().numpy(),4)}\nMSE                : {np.round(mse.cpu().numpy(),4)}\nEMD                : {np.round(emd.cpu().numpy(),4)}\n')

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
        plt.show();
        
    return similarity_factors, similarity_kernels, similarity_means, mse, emd, emd_means

def plot_activation(world, weights, training_parameters):

    device = world.device
    world.device = 'cpu'
    world.SMs = world.SMs.to('cpu')
    world.opt.frs = world.opt.frs.to('cpu')
    
    kernel_size = (world.opt.N_SMs, world.opt.N_pre, world.opt.N_delays)
    generative_model_ae = WassA(kernel_size, output=training_parameters.output, do_bias=training_parameters.do_bias, device='cpu')
    input, output = world.draw_input(nb_trials=1)
    factors, reconstructed = generative_model_ae(input)
    padded_factors = torch.nn.functional.pad(factors, (kernel_size[2]//2,kernel_size[2]//2,0,0,0,0), mode='constant')
    fig_raster, ax_raster = plot_colored_raster(input, output,world.opt.N_delays);
    for k in range(kernel_size[0]):
        ax_raster.plot((kernel_size[1]/factors.max().item())*padded_factors[0,k].detach().cpu().numpy())
    
    print(f'Maximum value for activations: {factors.max()}')

    world.device = device
    world.SMs = world.SMs.to(device)
    world.opt.frs = world.opt.frs.to(device)
    
    return reconstructed

def plot_robustness(results_mse, results_emd, coefs, noise_type, metric_name, metrics_labels, logplot=False, beta_fit=False):
    
    fig, ax = plt.subplots()
    
    if noise_type=='spontaneous':
        xlabel='Percentage of background noise'
        title='Robustness to background noise'
        coefs = coefs*100
    elif noise_type=='jitter':
        xlabel='Standard deviation of temporal jitter (in a.u.)'
        title='Robustness to temporal jitter'
    elif noise_type=='dropout':
        xlabel='Dropout probability' 
        title='Robustness to probabilistic dropout of the neurons'
        
    ind_m = metrics_labels.index(metric_name)
    ylabel = metrics_labels[ind_m]

    if beta_fit:
        bottom_mse, bottom_emd, top_mse, top_emd = np.zeros([len(coefs)]), np.zeros([len(coefs)]), np.zeros([len(coefs)]), np.zeros([len(coefs)])
        q = [0.05,0.95]
        for i in range(len(coefs)):
            paramz = beta.fit(results_mse[:,i,ind_m]*.9999+.00001, floc=0, fscale = 1)
            bottom_mse[i], top_mse[i] = beta.ppf(q, a=paramz[0], b=paramz[1])
            paramz = beta.fit(results_emd[:,i,ind_m]*.9999+.00001, floc=0, fscale = 1)
            bottom_emd[i], top_emd[i] = beta.ppf(q, a=paramz[0], b=paramz[1])
    else:
        bottom_mse = results_mse[:,:,ind_m].mean(axis=0)-results_mse[:,:,ind_m].std(axis=0)
        bottom_emd = results_emd[:,:,ind_m].mean(axis=0)-results_emd[:,:,ind_m].std(axis=0)
        top_mse = results_mse[:,:,ind_m].mean(axis=0)+results_mse[:,:,ind_m].std(axis=0)
        top_emd = results_emd[:,:,ind_m].mean(axis=0)+results_emd[:,:,ind_m].std(axis=0)

    ax.plot(coefs, results_mse[:,:,ind_m].mean(axis=0), '.',color='darkolivegreen', label='MSE')
    ax.plot(coefs, results_emd[:,:,ind_m].mean(axis=0), '.',color='b', label='EMD') 
    ax.fill_between(coefs, bottom_mse, top_mse, facecolor='darkolivegreen', edgecolor=None, alpha=.3)
    ax.fill_between(coefs, bottom_emd, top_emd, facecolor='b', edgecolor=None, alpha=.3)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=20)
    #ax.set_xlim([0,90])
    ax.legend(fontsize=12);

def plot_results_3D(results,variable_labels,metrics_labels,x_label,x_values,y_label,y_values,log_x=False,log_y=False,metric_name='factors similarity'):
    
    ind_x = variable_labels.index(x_label)
    ind_y = variable_labels.index(y_label)
    
    axes = np.delete(np.arange(len(variable_labels)),[ind_x,ind_y]).tolist()
    
    mean_results = results.mean(axis=axes)
    std_results = results.std(axis=axes)
    
    ind_m = metrics_labels.index(metric_name)
    
    metric = metrics_labels[ind_m]
    ax = plt.figure(figsize=(10,10)).add_subplot(projection='3d')

    if log_x: x_values = np.log10(x_values)
    if log_y: y_values = np.log10(y_values)
    
    X, Y = np.meshgrid(y_values,x_values)
    
    if ind_x>ind_y:
        mean_results = mean_results.swapaxes(0,1)

    # Plot the 3D surface
    ax.plot_surface(X, Y, mean_results[:,:,ind_m], edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.contourf(X, Y, mean_results[:,:,ind_m], zdir='z', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, mean_results[:,:,ind_m], zdir='x', offset=0, cmap='coolwarm')
    ax.contourf(X, Y, mean_results[:,:,ind_m], zdir='y', offset=x_values[-1], cmap='coolwarm')

    if ind_m<2:
        max_mean = torch.max(mean_results[:,:,ind_m])
    else:
        max_mean = torch.min(mean_results[:,:,ind_m])

    ax.set(xlim=(y_values[0], y_values[-1]), ylim=(x_values[0], x_values[-1]), zlim=(0, torch.max(mean_results[:,:,ind_m])),
           xlabel=y_label, ylabel=x_label, zlabel=metric)

    x_max_mean_1, x_max_mean_2 = (mean_results[:,:,ind_m]==max_mean).nonzero()[0]

    print(f'Best score for {metric} is {max_mean} with {x_label} = {x_values[x_max_mean_1]} - {y_label} = {y_values[x_max_mean_2]}')
    
def plot_results_2D(results_mse,results_emd,metrics_labels,x_label,x_values,metric_name='factors similarity',beta_fit=False):
    
    fig, ax = plt.subplots()
        
    ind_m = metrics_labels.index(metric_name)

    if beta_fit:
        bottom_mse, bottom_emd, top_mse, top_emd = np.zeros([len(x_values)]), np.zeros([len(x_values)]), np.zeros([len(x_values)]), np.zeros([len(x_values)])
        q = [0.05,0.95]
        for i in range(len(x_values)):
            paramz = beta.fit(results_mse[:,i,ind_m]*.9999+.00001, floc=0, fscale = 1)
            bottom_mse[i], top_mse[i] = beta.ppf(q, a=paramz[0], b=paramz[1])
            paramz = beta.fit(results_emd[:,i,ind_m]*.9999+.00001, floc=0, fscale = 1)
            bottom_emd[i], top_emd[i] = beta.ppf(q, a=paramz[0], b=paramz[1])
    else:
        bottom_mse = results_mse[:,:,ind_m].mean(axis=0)-results_mse[:,:,ind_m].std(axis=0)
        bottom_emd = results_emd[:,:,ind_m].mean(axis=0)-results_emd[:,:,ind_m].std(axis=0)
        top_mse = results_mse[:,:,ind_m].mean(axis=0)+results_mse[:,:,ind_m].std(axis=0)
        top_emd = results_emd[:,:,ind_m].mean(axis=0)+results_emd[:,:,ind_m].std(axis=0)

    ax.plot(x_values, results_mse[:,:,ind_m].mean(axis=0), '.',color='darkolivegreen', label='MSE')
    ax.plot(x_values, results_emd[:,:,ind_m].mean(axis=0), '.',color='b', label='EMD') 
    ax.fill_between(x_values, bottom_mse, top_mse, facecolor='darkolivegreen', edgecolor=None, alpha=.3)
    ax.fill_between(x_values, bottom_emd, top_emd, facecolor='b', edgecolor=None, alpha=.3)

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    #ax.set_title(title, fontsize=20)
    #ax.set_xlim([0,90])
    ax.legend(fontsize=12);
    
def plot_results_std(ax, results, coefs, xlabel, ylabel, legend, color, ymax=None, ymin=None, logplot=False):
     
    mean_, std_ = results.mean(axis=0), results.std(axis=0)

    if ymin is not None:
        bottom_ = np.maximum(mean_ - std_, ymin*np.ones([len(mean_)]))
    else:
        bottom_ = mean_ - std_
    if ymax is not None:
        top_= np.minimum(mean_ + std_, ymax*np.ones([len(mean_)]))
    else:
        top_ = mean_ + std_

    if logplot:
        ax.semilogx(coefs, mean_, '.',color=color, label=legend)
    else:
        ax.plot(coefs, mean_, '.',color=color, label=legend)
    ax.fill_between(coefs, bottom_, top_, facecolor=color, edgecolor=None, alpha=.3)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=12);
    return ax