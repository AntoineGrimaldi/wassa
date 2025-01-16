import torch, os, matplotlib
import matplotlib.pyplot as plt

def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))
    
class sm_generative_model:
    def __init__(self, dataset_parameters, device='cpu'):
        
        self.opt = dataset_parameters()
        self.device = device
        torch.manual_seed(self.opt.seed)
        # initialization of the different spiking motifs probability distributions
        self.SMs = torch.zeros(self.opt.N_SMs, self.opt.N_pre, self.opt.N_delays, device = device)
        # average probability of having a spike for a specific timestep (homogeneous Poisson)
        self.opt.frs = self.opt.frs.to(device)
        proba_timestep = self.opt.frs*1e-3
        # compute the average number of spike per motif for each neuron
        N_spikes_per_motif = proba_timestep*self.opt.N_delays
        self.spike_times = []
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
                spike_times = torch.randint(int(self.opt.temporal_jitter), int(self.opt.N_delays-self.opt.temporal_jitter), [int(torch.round(N_spikes_per_motif[n]))])
                self.spike_times.append((k,n,spike_times))
                for s in range(len(spike_times)):
                    if self.opt.temporal_jitter>0:
                        self.SMs[k, n] += gaussian_kernel(self.opt.N_delays, spike_times[s], self.opt.temporal_jitter).to(device)
                    else:
                        self.SMs[self.spike_times[s]] = 1
        
        # normalize kernels to have each row suming to 1 (probability distribution)
        self.SMs.div_(torch.norm(self.SMs, p=1, dim=-1, keepdim=True)+1e-14)
    
    def draw_input(self, nb_trials=10):
        
        kernels = (1-self.opt.additive_noise)*self.SMs+self.opt.additive_noise*torch.ones_like(self.SMs)/self.opt.N_delays
        
        # initialize input and output tensors
        input_proba = -1*torch.ones(nb_trials, self.opt.N_pre, self.opt.N_timesteps, device=self.device)
        output_rp = torch.zeros(nb_trials, self.opt.N_SMs, self.opt.N_timesteps, device=self.device)
        
        nb_motifs = torch.round(nb_trials*self.opt.N_timesteps*self.opt.freq_sms*1e-3)
        if self.opt.overlapping_sms:
        # iterate over the different occurrences of motifs to modify the local distribution
            for k in range(self.opt.N_SMs):
                nb_motif_k = int(nb_motifs[k])
                for n in range(nb_motif_k):
                    trial = torch.randint(nb_trials, [1])
                    time = torch.randint(self.opt.N_delays//2, self.opt.N_timesteps-(self.opt.N_delays//2+1), [1])                    
                    previous = input_proba[trial,:,time-(self.opt.N_delays//2):time+self.opt.N_delays//2+1].squeeze(0)
                    new = torch.zeros_like(previous, device=self.device)
                    # get dropped indices
                    if self.opt.dropout_proba:
                        not_to_keep = torch.where(torch.bernoulli(torch.ones_like(self.ind_inv)*self.opt.dropout_proba)==1)[0]
                        not_kept_ind = self.ind_inv[not_to_keep]
                        motif = kernels[k]
                        motif[not_kept_ind] = 1/self.opt.N_delays
                    else:
                        motif = kernels[k]
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
                    motif = kernels[k].squeeze(0)
                    motif[not_kept_ind] = 1/self.opt.N_delays
                else:
                    motif = kernels[k]
                input_proba[trial,:,time-(self.opt.N_delays//2):time+self.opt.N_delays//2+1] = motif
                output_rp[trial,k,time] = 1
                nb_added += 1
                loc_grid[trial,poss_loc_time[ind_loc]] = 0

        # random distribution when no modification was added
        input_proba[input_proba==-1] = 1/self.opt.N_delays
        # normalizing to required firing rates
        input_proba = input_proba.div_(torch.norm(input_proba, p=1, dim=-1, keepdim=True))*(self.opt.frs.unsqueeze(0).unsqueeze(-1).repeat(nb_trials,1,self.opt.N_timesteps)*1e-3*self.opt.N_timesteps)
        # harsh threshold to have a single spike in one timebin
        input_proba[input_proba>1] = 1
        # Bernoulli trial on the probability distribution
        input_rp = torch.bernoulli(input_proba)
                    
        return input_rp, output_rp

def generate_dataset(parameters, record_path='../synthetic_data/', num_samples=None, verbose=True,  device='cpu'):

    if not os.path.exists(record_path):
        os.mkdir(record_path)
        
    if num_samples is not None:
        num_train, num_test = num_samples
    else:
        # divide into train (50%) and test (50%) sets
        num_train = num_test = parameters.N_samples//2
    model_path = record_path+f'generative_model_'+parameters().get_parameters()
    trainset_path = record_path+f'synthetic_rp_trainset_{num_train}_'+parameters().get_parameters()+'.pt'
    testset_path = record_path+f'synthetic_rp_testset_{num_test}_'+parameters().get_parameters()+'.pt'
    if os.path.exists(model_path):
        #torch.serialization.add_safe_globals([sm_generative_model])
        sm = torch.load(model_path, map_location=device)#, weights_only = True)
    else:
        sm = sm_generative_model(parameters, device=device)
        torch.save(sm, model_path)

    for dataset in ['trainset', 'testset']:
        if dataset=='trainset':
            num = num_train
        elif dataset=='testset':
            num = num_test
        dataset_path = record_path+f'synthetic_rp_{dataset}_{num}_'+parameters().get_parameters()+'.pt'
        if verbose: print(dataset_path)
        if os.path.exists(dataset_path):
            dataset_input_list, dataset_output_list = torch.load(dataset_path, map_location=device)#, weights_only = True)
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


    