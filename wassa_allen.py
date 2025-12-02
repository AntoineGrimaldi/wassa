import torch, glob
import matplotlib.pyplot as plt
import numpy as np
from wassa_plots import plot_SM, plot_results
from wassa_utils import get_best_results, correlation_kernels
from wassa_training import get_training_parameters, learn_motifs, unsupervised_learning
from wassa_metrics import get_similarity, torch_cdf_loss, WassDist, kernels_diff
from wassa import WassA
from dataset_generation import kfold_dataset, generate_dataset, get_dataset_parameters
from tqdm import tqdm 

def make_allen_dataset(dataset_path, number_samples_per_image=10, number_folds=5, device='cpu'):
    
    files_list = glob.glob(dataset_path)
    all_trainsets, all_testsets, all_othersets = [], [], []
    for f in range(len(files_list)):
        data = torch.load(files_list[f], weights_only = True)
        data = data.to(torch.float32).to(device)
        number_samples, number_neurons, number_timesteps = data.shape
        number_images = number_samples//number_samples_per_image
    
        for image in range(number_images):
            trainsets, testsets, indices = kfold_dataset(data[image*number_samples_per_image:(image+1)*number_samples_per_image],k=number_folds)
            other_testsets = []
            for i in range(len(testsets)):
                image_numbers = torch.randint(number_images,(testsets[0].shape[0],))
                while image in image_numbers:
                    image_numbers = torch.randint(number_images,(testsets[0].shape[0],))
                sample_numbers = torch.randint(number_samples_per_image,(testsets[0].shape[0],))
                other_testsets.append(data[image_numbers*number_samples_per_image+sample_numbers])
            all_trainsets += trainsets
            all_testsets += testsets
            all_othersets += other_testsets
            
    return all_trainsets, all_testsets, all_othersets

def reconstruction_comparison(dataset_name,training_parameters,metric_names,frs=False,saving_path = 'results/allen_',device='cpu'):

    mse_loss = torch.nn.MSELoss()
    emd_loss = WassDist(zeros='same',normalize=training_parameters['normalize_input'])

    dataset_path = '../'+dataset_name+'/*'
    all_trainsets, all_testsets, all_othersets = make_allen_dataset(dataset_path)
    results = torch.zeros([len(all_trainsets),7,2])
    
    for ind in tqdm(range(len(all_trainsets))):

        if frs:
            training_set, testing_set, othersets = all_trainsets[ind].mean(dim=-1).unsqueeze(-1).to(device), all_testsets[ind].mean(dim=-1).unsqueeze(-1).to(device), all_othersets[ind].mean(dim=-1).unsqueeze(-1).to(device)
        else:
            training_set, testing_set, othersets = all_trainsets[ind].to(device), all_testsets[ind].to(device), all_othersets[ind].to(device)

        num_samples, num_neurons, num_timesteps = training_set.shape
        training_parameters['kernel_size'] = (1,num_neurons, num_timesteps)
        model = WassA(training_parameters,device=device)
        model_name = saving_path+dataset_name+str(ind)+get_training_parameters(training_parameters)
        model, training_metrics, _ = learn_motifs(model,training_set,testing_set,training_parameters,model_name,metric_names,None,verbose=False)
        if training_parameters['normalize_input']:
            othersets = torch.nn.functional.normalize(othersets, p=1, dim=2)
            testing_set = torch.nn.functional.normalize(testing_set, p=1, dim=2)
        factors_other, otherset_hat = model(othersets)
        factors_testing, testingset_hat = model(testing_set)
        results[ind,0,0] = mse_loss(testingset_hat,testing_set).detach()
        results[ind,0,1] = mse_loss(otherset_hat,othersets).detach()
        results[ind,1,1] = emd_loss(otherset_hat,othersets).detach()
        results[ind,1,0] = emd_loss(testingset_hat,testing_set).detach()
        results[ind,2,1] = (1-torch.var(othersets-otherset_hat)/torch.var(othersets)).nanmean().detach()
        results[ind,2,0] = (1-torch.var(testing_set-testingset_hat)/torch.var(testing_set)).nanmean().detach()
        correlation_kernel_testset, mse_kernel_testset, emd_kernel_testset = 0, 0, 0
        correlation_kernel_otherset, mse_kernel_otherset, emd_kernel_otherset = 0, 0, 0
        for i in range(othersets.shape[0]):
            input_spikes, input_spikes_others, weights = testing_set[i].unsqueeze(0).detach(), othersets[i].unsqueeze(0).detach(), model.weights.detach()
            correlation_kernel_testset += correlation_kernels(input_spikes, weights)
            correlation_kernel_otherset += correlation_kernels(input_spikes_others, weights)
            mse_kernel_testset += kernels_diff(input_spikes, weights, 'mse')
            mse_kernel_otherset += kernels_diff(input_spikes_others, weights, 'mse')
            emd_kernel_testset += kernels_diff(input_spikes, weights, 'emd')
            emd_kernel_otherset += kernels_diff(input_spikes_others, weights, 'emd')
        results[ind,3,0] = correlation_kernel_testset/othersets.shape[0]
        results[ind,3,1] = correlation_kernel_otherset/othersets.shape[0]
        results[ind,4,0] = mse_kernel_testset/othersets.shape[0]
        results[ind,4,1] = mse_kernel_otherset/othersets.shape[0]
        results[ind,5,0] = emd_kernel_testset/othersets.shape[0]
        results[ind,5,1] = emd_kernel_otherset/othersets.shape[0]
        results[ind,6,0] = factors_testing.detach().mean()
        results[ind,6,1] = factors_other.detach().mean()

    return results

dataset_name = 'allen_data_by_image'
device = 'cuda:0'

saving_path = 'results/wassa_allen_'
metric_names = ['training loss', 'testing loss', 'activations', 'explained variance']

training_parameters_emd = {
    'kernel_size' : (1, 1, 1),
    'loss_type' : 'emd',
    'activation' : 'emd like',
    'sigmoid' : False,
    'kernels_norm' : 2,
    'N_learnsteps' : 600,
    'learning_rate' : .01,
    'penalty_type' : [None],
    'lambda' : [0],
    'batch_size' : None,
    'do_bias' : False,
    'weight_init' : 'flat',
    'normalize_input' : True,
}

training_parameters_mse = {
    'kernel_size' : (1, 1, 1),
    'loss_type' : 'mse',
    'activation' : 'conv',
    'sigmoid' : False,
    'kernels_norm' : 2,
    'N_learnsteps' : 600,
    'learning_rate' : .05,
    'penalty_type' : [None],
    'lambda' : [0],
    'batch_size' : None,
    'do_bias' : False,
    'weight_init' : 'flat',
    'normalize_input' : True
}

#results_emd = reconstruction_comparison(dataset_name,training_parameters_emd,metric_names,saving_path=saving_path,device=device)
results_mse = reconstruction_comparison(dataset_name,training_parameters_mse,metric_names,saving_path=saving_path,device=device)
results_frs = reconstruction_comparison(dataset_name,training_parameters_mse,metric_names,frs=True,saving_path=saving_path,device=device)