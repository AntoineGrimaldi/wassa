import torch, os, time
from wassa.wassa import WassA
from wassa.wassa_metrics import WassDist, torch_cdf_loss, correlation_latent_variables, correlation_kernels, kernels_diff
from wassa.wassa_plots import plot_results
from tqdm import tqdm

def write_path(dir, date, world_params, training_params, iteration=0):
    return f'{dir}{date}_{world_params.get_parameters()}_{training_params.get_parameters()}_{iteration}'

def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))

def smoothing(x,smoothing_window_size):
    
    device = x.device
    N_batch, N_kernel, N_timesteps = x.shape
    #weights = gaussian_kernel(smoothing_window_size,smoothing_window_size//2,smoothing_window_size//10).unsqueeze(0).unsqueeze(0)/smoothing_window_size
    weights = torch.ones([1,1,smoothing_window_size], device=device)/smoothing_window_size
    flattened = torch.reshape(x.clone(), (N_batch*N_kernel, 1, N_timesteps))
    smoothed = torch.nn.functional.conv1d(flattened, weights, padding='same')
    return torch.reshape(smoothed, (N_batch, N_kernel, N_timesteps))

def learn_offline(criterion, model, input_raster_plot, training_parameters, path, verbose=True, device='cpu'):
    
    if os.path.isfile(path + '.pth'):
        if verbose: print(path)
        model.load_state_dict(torch.load(path + '.pth', map_location=torch.device(device)))
        LOSS = torch.load(path + '_loss.pth')
        model.decoding_weights.to(device)
    else:
        LOSS = []
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=training_parameters.learning_rate)

        if training_parameters.batch_size:
            nb_batch = torch.div(input_raster_plot.shape[0],training_parameters.batch_size,rounding_mode='floor')
            if nb_batch*training_parameters.batch_size!=input_raster_plot.shape[0]: nb_batch+=1
        else:
            nb_batch = 1
        
        for i_step in (tqdm(range(training_parameters.N_learnsteps)) if verbose else range(training_parameters.N_learnsteps)):

            # shuffle sample indices
            idx = torch.randperm(input_raster_plot.size(0))
            input_raster_plot = input_raster_plot[idx]
            
            for n in range(nb_batch):
                if training_parameters.batch_size:
                    X = input_raster_plot[n*training_parameters.batch_size:min((n+1)*training_parameters.batch_size,input_raster_plot.shape[0])]
                else:
                    X = input_raster_plot
                
                X.div_(X.sum(dim=-1, keepdim=True)+1e-14)
                optimizer.zero_grad()
                
                factors, reconstruction = model(X)
                reconstruction = reconstruction.div(reconstruction.sum(dim=-1, keepdim=True)+1e-14)
                loss = criterion(reconstruction, X)

                if training_parameters.lambda_ != 0:
                    if training_parameters.penalty_type == 'cc':
                        loss += training_parameters.lambda_*cross_correlation_comp(factors, mode = 'mean')
                    elif training_parameters.penalty_type == 'max_cc':
                        loss += training_parameters.lambda_*cross_correlation_comp(factors, mode = 'max')
                    elif training_parameters.penalty_type == 'smoothed_orthogonality':
                        smoothed_factors = smoothing(factors,training_parameters.smoothwind)
                        loss += training_parameters.lambda_*orthogonality(smoothed_factors)
                    elif training_parameters.penalty_type  == 'kernels_orthogonality':
                        loss += training_parameters.lambda_*kernels_orthogonality(model.decoding_weights)
                    elif training_parameters.penalty_type  == 'sparsity':
                        loss += training_parameters.lambda_*factors.abs().mean()
                    
                LOSS.append(loss.clone().detach().cpu().numpy())
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    model.decoding_weights.data.clamp_(min=0)
                    model.decoding_weights.data.div_((torch.linalg.norm(model.decoding_weights.data, ord=2, dim=(1,2), keepdim=True)+1e-14).repeat(1,model.decoding_weights.shape[1],model.decoding_weights.shape[2]))
        
        torch.save(model.state_dict(), path + '.pth')
        torch.save(LOSS, path + '_loss.pth')

    return model, LOSS

def cross_correlation_comp(factors, mode = 'mean'):
    
    device = factors.device
    
    N_batch, N_kernel, N_timesteps = factors.shape
    normalized_factors = factors.clone().div_(torch.norm(factors, p=2, dim=-1, keepdim=True)+1e-14)
    reshaped_factors = normalized_factors.reshape(N_batch*N_kernel,1,N_timesteps)
    padded_factors = torch.nn.functional.pad(reshaped_factors, (torch.div(N_timesteps,2,rounding_mode='floor'),torch.div(N_timesteps,2,rounding_mode='floor'),0,0,0,0), mode='constant')
    
    cross_corr = torch.nn.functional.conv1d(padded_factors, reshaped_factors)
    
    if mode == 'mean':
        reduced_cc = cross_corr.mean(dim=-1)
    elif mode == 'max':
        reduced_cc = cross_corr.amax(dim=-1)

    blocks = torch.ones([N_kernel,N_kernel],device=device).triu(diagonal=1)
    bloc_diag = torch.block_diag(*blocks.unsqueeze(0).repeat(N_batch,1,1))
    
    valid_cross_corr = reduced_cc*bloc_diag

    return valid_cross_corr.mean()

def kernels_orthogonality(weights):

    n_kernels, n_neurons, n_delays = weights.shape

    padded_weights = torch.nn.functional.pad(weights,(torch.div(n_delays,2,rounding_mode='floor'),torch.div(n_delays,2,rounding_mode='floor'),0,0,0,0), mode='constant')
    cross_correlation = torch.nn.functional.conv1d(padded_weights,weights)
    max_cross_correlation = cross_correlation.amax(dim=-1).T

    return max_cross_correlation.triu(diagonal=1).mean()

def orthogonality(factors):
    device = factors.device
    N_batch, N_kernel, N_timesteps = factors.shape
    normalized_factors = factors.clone().div_(torch.norm(factors, p=1, dim=-1, keepdim=True)+1e-14)
    correlation_matrix = torch.zeros([N_kernel,N_kernel], device=device)
    for b in range(N_batch):
        correlation_matrix += torch.corrcoef(normalized_factors[b])
    return (correlation_matrix.triu().sum()-torch.trace(correlation_matrix))/(N_batch*N_kernel*N_timesteps)

def train_and_plot(dataset_parameters, sm, trainset_input, testset_input, testset_output, training_parameters, date, results_directory = '../results/', iteration = 0, plot = False, order_sms=True, verbose = False, device = 'cpu'):

    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    autoencoder = WassA(training_parameters[0].kernel_size, weight_init = training_parameters[0].weight_init, output=training_parameters[0].output, do_bias=training_parameters[0].do_bias, device=device)
    loss_evolution = []
    for params in training_parameters:
        if params.loss_type == 'mse':
            loss = torch.nn.MSELoss()
        else:
            loss = WassDist(p=params.wass_order, zeros=params.zeros)
        if len(training_parameters)==2:
            path = write_path(results_directory, date+'_combined', dataset_parameters, params, iteration=iteration)
            learnsteps_init = params.N_learnsteps
            params.N_learnsteps = learnsteps_init//2
        else:
            path = write_path(results_directory, date, dataset_parameters, params, iteration=iteration)
        autoencoder, loss_over_epochs = learn_offline(loss, autoencoder, trainset_input, params, path, verbose=verbose, device = device)
        loss_evolution.append(loss_over_epochs)
        
        if len(training_parameters)==2: params.N_learnsteps = learnsteps_init

    similarity_factors, similarity_kernels, similarity_means, mse, emd, emd_means = plot_results(sm, loss_evolution[-1], autoencoder, testset_input, testset_output, order_sms = order_sms, plot = plot, verbose = verbose, device = device)
            
    return similarity_factors, similarity_kernels, similarity_means, mse, emd, emd_means, loss_evolution, autoencoder