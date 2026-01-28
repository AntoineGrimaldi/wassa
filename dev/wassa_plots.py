import matplotlib, torch, time
import matplotlib.pyplot as plt
import numpy as np
from wassa_metrics import get_similarity
from wassa_utils import sliding_correlation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from IPython import display

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_results(training_metrics, metrics_names, trained_layer_of_neurons, input_raster_plot, true_occurence=None, true_weights=None, moving_window = 50, order_sms = False, warping_coef = 1, verbose = True, device = 'cpu'):

    cmap = matplotlib.colormaps['Set3']
    cmap_rec = matplotlib.colormaps['Purples']

    # print the different losses
    fig_loss, ax_loss = fig, ax = plot_training_metrics(training_metrics,metrics_names)
        
    figsize_kernels = (12.75,2)
    if trained_layer_of_neurons.tied_weights:
        N_SMs, N_pre, N_delays = trained_layer_of_neurons.weights.shape
        weights = trained_layer_of_neurons.weights.data.detach().cpu()
        plot_SM(weights, figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);
    else:
        N_SMs, N_pre, N_delays = trained_layer_of_neurons.decoder[0].weight.shape
        decoding_weights = trained_layer_of_neurons.decoder[0].weight.data.detach().cpu()
        encoding_weights = trained_layer_of_neurons.encoder[0].weight.data.detach().cpu()
        plot_SM(encoding_weights, figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);
        plot_SM(decoding_weights, figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);
    if true_weights is not None:
        plot_SM(true_weights, figsize=figsize_kernels, N_show = N_SMs, order_sms = order_sms);

    factors, reconstruction = trained_layer_of_neurons(input_raster_plot)
    N_sample, _, _ = input_raster_plot.shape
    random_ind = torch.randint(N_sample,[1])
    if true_occurence is not None:
        fig_raster, ax_raster = plot_colored_raster(input_raster_plot[random_ind].squeeze(0), true_occurence[random_ind].squeeze(0).cpu().numpy(), N_delays, warping_coef);
    else:
        fig_raster, ax_raster = plot_raster(input_raster_plot[random_ind].squeeze(0));
    padded_factors = torch.nn.functional.pad(factors[random_ind], (N_delays//2,N_delays//2, 0, 0), mode='constant')
    for k in range(N_SMs):
        ax_raster.plot((N_pre/padded_factors.max().item())*padded_factors[0,k].detach().cpu().numpy())
    plt.show();

#TODO :  remove numpy to have only torch tensors 
def plot_colored_raster(input_rp, output_rp, N_delays, warping_coef, title = 'raster plot'):
    cmap_2 = matplotlib.colormaps['Set3']
    fig, ax = plot_raster(input_rp, title = title)
    indices = np.where(output_rp==1)
    for k in range(len(indices[0])):
        ax.axvspan(indices[1][k]-N_delays*warping_coef//2, indices[1][k]+N_delays*warping_coef//2+1, facecolor=cmap_2(indices[0][k]), alpha=0.5)
    return fig, ax

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

def plot_SM(SMs, N_show = 5, order_sms=False, cmap='Purples', colors=None, aspect=None, figsize = (12, 1.61803)):
    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)

    SMs = SMs.to('cpu')

    cmap_2 = matplotlib.colormaps['Set1']
    
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
        ax.set_xlabel('spike latency')
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
        axs[0].set_ylabel('neuron address')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs[:], orientation='vertical', ticks=[0, 1],
                format=matplotlib.ticker.FixedFormatter(np.round([SMs.min().item(), SMs.max().item()],3)))
    else:
        axs.set_ylabel('neuron address')
        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=axs, orientation='vertical', ticks=[0, 1],
                format=matplotlib.ticker.FixedFormatter(np.round([SMs.min().item(), SMs.max().item()],3)))

    return fig, axs

def plot_hp_tuning(results, first_hp, second_hp, saved_hyperparameters, losses, metrics, fixed_hp = None, fixed_ind = [0], logplot = True, ymax = 1, ymin = 0, results_allen=True):

    hyperparameters_names = list(saved_hyperparameters.keys())

    if results_allen:
        local_results = results.clone().nanmean(axis=-2).swapaxes(-1,-2)
    else:
        local_results = results.clone()
    
    ind_1, ind_2 = hyperparameters_names.index(first_hp), hyperparameters_names.index(second_hp)
    
    if ind_1>ind_2:
        local_results = local_results.swapaxes(ind_1,ind_2)
        hyperparameters = list(saved_hyperparameters.items())
        hyperparameters[ind_1], hyperparameters[ind_2] = hyperparameters[ind_2], hyperparameters[ind_1]
        saved_hyperparameters = dict(hyperparameters)
        hyperparameters_names = list(saved_hyperparameters.keys())
        ind_2, ind_1 = ind_1, ind_2

    if fixed_hp:
        for ind_hp, hp in enumerate(fixed_hp):
            ind_f = hyperparameters_names.index(hp)
            print(f'{hp} is {saved_hyperparameters[hyperparameters_names[hyperparameters_names.index(hp)]][fixed_ind[ind_hp]]}')
            local_results = local_results.unsqueeze(0).swapaxes(0,ind_f+1)[fixed_ind[ind_hp]]
            #if ind_f<ind_1: ind_1 -= 1
            #if ind_f<ind_2: ind_2 -= 1

    # to bring the number of iter ind=-1 to the front 
    local_results = local_results.unsqueeze(0).swapaxes(0,-1).squeeze(-1)
    ind_1 += 1
    ind_2 += 1
    local_results = local_results.swapaxes(ind_2,-3)
    local_results = local_results.swapaxes(ind_1,0)
    
    colors = ['darkolivegreen','blue', 'orangered']
    fig, ax = plt.subplots(len(metrics),len(saved_hyperparameters[second_hp]), figsize = (15,10))
    for ind_scnd, lambda_ in enumerate(saved_hyperparameters[second_hp]):
        for ind_m, metric in enumerate(metrics):
            for ind_l, loss in enumerate(losses):
                results_plot = local_results[...,ind_scnd,ind_l,ind_m].flatten(start_dim=1)
                xlabel = f'{first_hp} for\n{second_hp} is {lambda_}'
                ylabel = metrics[ind_m]
                if first_hp == 'penalty_type' or first_hp == 'lambda':
                    coefs = [saved_hyperparameters[first_hp][i][0] for i in range(len(saved_hyperparameters[first_hp]))]
                else:
                    coefs = saved_hyperparameters[first_hp]
                if len(saved_hyperparameters[second_hp])>1:
                    ax[ind_m,ind_scnd] = plot_results_std(ax[ind_m,ind_scnd],results_plot.T,coefs,xlabel,ylabel,loss,colors[ind_l],logplot = logplot,ymax=ymax,ymin=ymin)
                else:
                    ax[ind_m] = plot_results_std(ax[ind_m],results_plot.T,coefs,xlabel,ylabel,loss,colors[ind_l],logplot = logplot,ymax=ymax,ymin=ymin)

def plot_training_metrics(training_metrics,metric_names,i_step=-1):
    fig, ax = plt.subplots(ncols=len(metric_names),figsize=(12,3))
    for i in range(len(metric_names)):
        ax[i].plot(training_metrics[:i_step,i])
        ax[i].set_title(metric_names[i])
    return fig, ax

def online_monitoring(training_metrics,metric_names,i_step,start_time):
    # TODO: simplify it and avoid the flickering
    elapsed_time = time.time()-start_time
    remaining_time = elapsed_time/(1+i_step)*(training_metrics.shape[0]-1-i_step)
    remaining_time_hour = int(remaining_time//3600)
    remaining_time_min = int((remaining_time-remaining_time_hour*3600)//60)
    fig, ax = plot_training_metrics(training_metrics,metric_names,i_step=i_step)
    fig.suptitle(f'Iteration #{i_step} - estimated remaining time: {remaining_time_hour}h{remaining_time_min}m',y=1.1)
    plt.close("all")
    display.clear_output(wait=True)
    display.display(fig)

def plot_similarity_matrix(corr_kernels_value, n_iter, n_folds, n_variables, quant=.99):
    
    quantile_value = torch.quantile(corr_kernels_value,quant)
    corr_kernels_value[corr_kernels_value<quantile_value]=0

    cmap = matplotlib.colormaps['Set2']
    fig, ax = plt.subplots(figsize=(10,10))
    sim = ax.imshow(corr_kernels_value)
    for k in range(n_folds):
        for i in range(n_iter):
            for j in range(n_iter):
                rect1 = matplotlib.patches.Rectangle((n_variables*k+i*n_variables*n_folds-.5,n_variables*k+j*n_variables*n_folds-.5),n_variables, n_variables, color=cmap(k), fc = 'none',lw = 1)
                ax.add_patch(rect1)
                if i>0:
                    ax.hlines(n_variables*n_folds*i-.5,-.5,n_variables*n_folds*n_iter-.5,'r')
                if j>0:
                    ax.vlines(n_variables*n_folds*i-.5,-.5,n_variables*n_folds*n_iter-.5,'r')
    axins = inset_axes(
        ax,
        width="2%",  # width: 5% of parent_bbox width
        height="100%",  # height: 50%
        loc="lower left",
        bbox_to_anchor=(1.05, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    fig.colorbar(sim,cax=axins)

def plot_results_std(ax, results, coefs, metric_names, xlabel, metric, legend, color, do_ylabel, ymax=None, ymin=None, do_legend=False, logplot=False, quantile=False):

    results = results[...,metric_names.index(metric)].clone().cpu()

    mean_, std_ = results.mean(axis=(1,2)), results.std(axis=(1,2))
    
    if quantile:
        q5 = np.quantile(results,.2,axis=(1,2))
        q95 = np.quantile(results,.8,axis=(1,2))

    if quantile:
        bottom_ = q5
    elif ymin is not None:
        bottom_ = np.maximum(mean_ - std_, ymin*np.ones([len(mean_)]))
    else:
        bottom_ = mean_ - std_
        
    if quantile:
        top_ = q95
    elif ymax is not None:
        top_= np.minimum(mean_ + std_, ymax*np.ones([len(mean_)]))
    else:
        top_ = mean_ + std_

    if logplot:
        ax.semilogx(coefs, mean_, '.',color=color, label=legend)
    else:
        ax.plot(coefs, mean_, 'P',color=color, markeredgecolor='white', markersize=10, label=legend)

    #ax.scatter(coefs.unsqueeze(0).repeat(results.shape[0],1),results, color=color,alpha=.2)
    
    ax.fill_between(coefs, bottom_, top_, facecolor=color, edgecolor=None, alpha=.3)

    ax.set_ylim(ymin,ymax)
    
    if xlabel: 
        ax.set_xlabel(xlabel, fontsize=14)
    else: 
        ax.set_xticks([])
    if do_ylabel:
        ax.set_ylabel('similarity value', fontsize=14)
    else: 
        ax.set_yticks([])
    ax.set_title(metric, fontsize=16)
    if do_legend: 
        ax.legend(fontsize=12);

    return ax