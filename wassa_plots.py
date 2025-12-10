import numpy as np
import matplotlib, torch, time
import matplotlib.pyplot as plt
from IPython import display
from wassa_utils import estimate_spike_times

def plot_results(training_metrics, metrics_names, trained_layer_of_neurons, input_raster_plot, model=None, order_sms = False, spike_times = None):

    cmap = matplotlib.colormaps['Set3']
    cmap_rec = matplotlib.colormaps['Purples']

    fig_loss, ax_loss = plot_training_metrics(training_metrics,metrics_names)
        
    figsize_kernels = (12.75,2)
    if model is not None:
        #fig, ax, sorted_indices = plot_spike_times(model, order_sms=order_sms, figsize = figsize_kernels);
        plot_SM(model.SMs.detach().cpu(), figsize=figsize_kernels);
        sorted_indices = torch.arange(trained_layer_of_neurons.weights.shape[1])
    else:
        sorted_indices = torch.arange(trained_layer_of_neurons.weights.shape[1])

    N_SMs, N_pre, N_delays = trained_layer_of_neurons.weights.shape
    weights = trained_layer_of_neurons.weights.data.detach().cpu().numpy()
    plot_SM(weights, figsize=figsize_kernels, order_indices = sorted_indices, spike_times = spike_times);

def plot_training_metrics(training_metrics,metric_names,i_step=-1):
    fig, ax = plt.subplots(ncols=len(metric_names),figsize=(12,3))
    for i in range(len(metric_names)):
        ax[i].plot(training_metrics[:i_step,i])
        ax[i].set_title(metric_names[i])
    return fig, ax

def plot_raster(fig, ax, raster, color = 'black', alpha = 1, spikelength=.9, linewidths=2.0, xticks = 6, subplotpars=matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)):

    N_neurons, N_timesteps = raster.shape
    raster = raster.to('cpu')
    
    for i in range(0, N_neurons):
        ax.eventplot(np.where(raster[i, :] == 1.)[0], 
            colors=color,  alpha = alpha, lineoffsets=1.*i+spikelength/2, 
            linelengths=spikelength, linewidths=linewidths)

    ax.set_ylabel('Neurons')
    ax.set_xlabel('Time (a. u.)')
    ax.set_xlim(0, N_timesteps)
    ax.set_ylim(0, N_neurons)

    ax.set_yticks(np.arange(0, N_neurons, 1)+.5)
    ax.set_yticklabels('')
    for side in ['top', 'right']: ax.spines[side].set_visible(False)

    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(N_timesteps/4))
    ax.set_xticks(np.linspace(1, N_timesteps, xticks, endpoint=True))
    ax.set_xticklabels(np.linspace(1, N_timesteps, xticks, endpoint=True).astype(int))
    ax.set_title('Raster plot')

    return fig, ax

def plot_random_trial(raster_plot):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    cmap = matplotlib.colormaps['tab10']
    trial_number = torch.randint(raster_plot.shape[1],[1])
    for kernel in range(raster_plot.shape[0]-1):
        fig, ax = plot_raster(fig,ax,raster_plot[kernel,trial_number].squeeze(0),color=cmap(kernel));
    fig, ax = plot_raster(fig,ax,raster_plot[-1,trial_number].squeeze(0));
    plt.show();

def plot_averaged(raster_plot):
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    cmap = matplotlib.colormaps['tab10']
    avg_raster = raster_plot.mean(axis=1)
    values = torch.unique(avg_raster)
    max_value = values.max().item()
    for value in values:
        raster = avg_raster==value
        for kernel in range(raster_plot.shape[0]-1):
            fig, ax = plot_raster(fig,ax,raster[kernel].squeeze(0),color=cmap(kernel),alpha=value.item()/max_value);
        fig, ax = plot_raster(fig,ax,raster[-1].squeeze(0),alpha=value.item());
    plt.show();

def plot_spike_times(model, show_max = None, order_sms=False, color='black', colors=None, aspect=None, figsize = (12, 1.61803)):
    
    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)
    cmap_2 = matplotlib.colormaps['tab10']
    
    N_sms = len(model.spike_times)
    steps_pre = model.opt['N_pre']/10
    if show_max is not None:
        N_show = min(show_max, N_sms)
    else:
        N_show = N_sms
        
    fig, axs = plt.subplots(1, N_show, figsize=figsize, subplotpars=subplotpars)
    for i_SM in range(N_show):
        if N_show>1:
            ax = axs[i_SM]
            axs[0].set_ylabel('neuron adresse')
        else:
            ax = axs
            ax.set_ylabel('neuron adresse')
        ax.set_axisbelow(True)
        if order_sms:
            sorted_times, sorted_indices = torch.tensor(model.spike_times[i_SM])[:,1].sort()
            ax.eventplot(sorted_times.unsqueeze(1),color=color)
        else:
            sorted_indices = None
            ax.eventplot(torch.tensor(model.spike_times[i_SM])[:,1].unsqueeze(1),color=color)
        ax.set_xlim(0, model.opt['N_timesteps'])
        ax.set_xlabel('spike latency')
        t = ax.text(.1*model.opt['N_timesteps'], .95*model.opt['N_pre'], f'#{i_SM+1}', color='k' if colors is None else colors[i_SM])
        t.set_bbox(dict(facecolor=cmap_2(i_SM), edgecolor='black'))
        if not aspect is None: ax.set_aspect(aspect)

        ax.set_ylim(0, model.opt['N_pre'])
        ax.set_yticks(np.arange(0, model.opt['N_pre'], 1)+.5)
        ax.set_yticklabels('')

        for side in ['top', 'right']: ax.spines[side].set_visible(False)
        ax.set_xticks([1, model.opt['N_timesteps']//3, (model.opt['N_timesteps']*2)//3])
        ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(model.opt['N_timesteps']//4))
    return fig, axs, sorted_indices

def plot_stats(raster_plot, motifs_occurence, dataset_parameters):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].bar(np.arange(dataset_parameters['N_pre']),raster_plot.mean(axis=(0,1,3))/dataset_parameters['N_pre']*1e3)
    ax[0].set_xlabel('@ Neuron')
    ax[0].set_ylabel('Firing rate (in Hz)')
    ax[0].set_title('Average number of spikes per neuron');

    ax[1].bar(np.arange(dataset_parameters['N_sms']),motifs_occurence.mean(axis=(0)))
    ax[1].plot(dataset_parameters['proba_sms'], 'ro', label='expected number')
    ax[1].set_xlabel('Spiking motif #')
    ax[1].set_ylabel('Number of occurence')
    ax[1].set_title('Occurence of the different motifs');
    ax[1].legend();

def plot_SM(SMs, show_max = None, order_indices = None, spike_times = None, cmap='Purples', colors=None, aspect=None, figsize = (12, 1.61803)):
    
    subplotpars = matplotlib.figure.SubplotParams(left=0.125, right=.95, bottom=0.25, top=.975, wspace=0.05, hspace=0.05,)
    cmap_2 = matplotlib.colormaps['Set1']
    
    N_SMs, N_pre, N_delays = SMs.shape
    steps_pre = N_pre/10
    
    if show_max is not None:
        N_show = min(show_max, N_SMs)
    else:
        N_show = N_SMs
        
    fig, axs = plt.subplots(1, N_show, figsize=figsize, subplotpars=subplotpars)
    for i_SM in range(N_show):
        if N_show>1:
            ax = axs[i_SM]
        else:
            ax = axs
        ax.set_axisbelow(True)
        if order_indices is not None:
            ordered_sm = SMs[i_SM,order_indices]
            ax.pcolormesh(ordered_sm, cmap=cmap, vmin=SMs.min(), vmax=SMs.max())
            if spike_times is not None:
                ax.eventplot(torch.tensor(spike_times[i_SM])[order_indices].unsqueeze(1),color='red')
        else:
            ax.pcolormesh(SMs[i_SM], cmap=cmap, vmin=SMs.min(), vmax=SMs.max())
            if spike_times is not None:
                ax.eventplot(torch.tensor(spike_times[i_SM]).unsqueeze(1),color='red')
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

def plot_training_metrics(training_metrics,metric_names,i_step=-1):
    fig, ax = plt.subplots(ncols=len(metric_names),figsize=(12,3))
    for i in range(len(metric_names)):
        ax[i].plot(training_metrics[:i_step,i])
        ax[i].set_title(metric_names[i])
    return fig, ax

def online_monitoring(weights,training_metrics,metric_names,i_step,start_time):
    # TODO: simplify it and avoid the flickering
    elapsed_time = time.time()-start_time
    remaining_time = elapsed_time/(1+i_step)*(training_metrics.shape[0]-1-i_step)
    remaining_time_hour = int(remaining_time//3600)
    remaining_time_min = int((remaining_time-remaining_time_hour*3600)//60)
    fig_, ax_ = plot_SM(weights)
    fig, ax = plot_training_metrics(training_metrics,metric_names,i_step=i_step)
    fig.suptitle(f'Iteration #{i_step} - estimated remaining time: {remaining_time_hour}h{remaining_time_min}m',y=1.1)
    fig_.suptitle('weights')
    plt.close("all")
    display.clear_output(wait=True)
    display.display(fig)
    display.display(fig_)


def plot_results_std(ax, results, coefs, metric_names, xlabel, metric, legend, color, ylabel, ymax=None, ymin=None, xmax=None, xmin=None, do_legend=False, log_x=False, log_y=False, quantile=False, plot_all_points=False, min_log_val = 1e-5):

    results = results[...,metric_names.index(metric)].clone().cpu()

    mean_, std_ = results.mean(axis=(1,2)), results.std(axis=(1,2))
    
    if quantile:
        q5 = np.quantile(results,.01,axis=(1,2))
        q95 = np.quantile(results,.99,axis=(1,2))

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

    if log_x and log_y:
        ax.loglog(coefs, mean_, 'P',color=color, markeredgecolor='white', markersize=10, label=legend)
    elif log_x:
        ax.semilogx(coefs, mean_, 'P',color=color, markeredgecolor='white', markersize=10, label=legend)
    elif log_y:
        if (mean_==0).sum()>0: print(f'zero values are set as {min_log_val}') 
        mean_[mean_==0] = min_log_val
        ax.plot(coefs, np.log10(mean_), 'P',color=color, markeredgecolor='white', markersize=10, label=legend)
    else:
        ax.plot(coefs, mean_, 'P',color=color, markeredgecolor='white', markersize=10, label=legend)
    if log_y:
        bottom_[bottom_==0] = min_log_val
        top_[top_==0] = min_log_val
        ax.fill_between(coefs, np.log10(bottom_), np.log10(top_), facecolor=color, edgecolor=None, alpha=.3)
    else:
        ax.fill_between(coefs, bottom_, top_, facecolor=color, edgecolor=None, alpha=.3)

    if plot_all_points:
        if log_y:
            results[results==0] = min_log_val
            points = np.log10(results.flatten(start_dim=1))
        else:
            points = results.flatten(start_dim=1)
        x_value = torch.tensor(coefs).unsqueeze(1).repeat(1,points.shape[1]).flatten()
        y_value = points.flatten()
        ax.scatter(x_value,y_value,marker='+',color=color,alpha=.2)
        
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    else: 
        ax.set_xticks([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=14)
    else: 
        ax.set_yticks([])
    #ax.set_title(metric, fontsize=16)
    if do_legend: 
        ax.legend(fontsize=12);
    return ax

def make_violin(ax,results,color,separation_coef=.5,offset=1):
    
    if len(results.shape)==2:
        violin_parts = ax.violinplot(results,positions=[offset+i*separation_coef for i in range(results.shape[1])],showmedians=True)
        
    elif len(results.shape)==1:
        violin_parts = ax.violinplot(results,positions=[offset],showmedians=True)
    else:
        print(f'shape of the results is {results.shape}')

    # Make all the violin statistics marks red:
    for partname in ('cbars','cmins','cmaxes','cmedians', 'bodies'):
        vp = violin_parts[partname]
        if partname=='bodies':
            for k in range(len(vp)):
                vp[k].set_alpha(.3)
                vp[k].set_facecolor(color)
        else:
            vp.set_edgecolor(color)
            vp.set_linewidth(1)
    return ax