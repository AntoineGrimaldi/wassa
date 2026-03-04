import torch, os, glob

def get_dataset_parameters(dataset_parameters,include_samples=False):

    name = ''
    for key in dataset_parameters.keys():
        if key=='N_samples':
            if include_samples:
                name += f'{dataset_parameters[key]}'
        elif torch.is_tensor(dataset_parameters[key]):
            for el in range(min(dataset_parameters[key].numel(),4)):
                name += f'{dataset_parameters[key].flatten()[el].item()}'
        else:
            name += f'{dataset_parameters[key]}'
    return name

def gaussian_kernel(n_steps, mu, std):
    if std>0:
        x = torch.arange(n_steps)
        output = torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))
    else:
        output = torch.zeros(n_steps)
        output[mu] = 1
    return output
    
class sm_generative_model:
    def __init__(self, dataset_parameters, device='cpu'):
        
        self.opt = dataset_parameters
        self.device = device
        torch.manual_seed(self.opt['seed'])
        self.SMs = torch.zeros([self.opt['N_sms'],self.opt['N_pre'],self.opt['N_timesteps']],device=device)
        self.spike_times = []
        for k in range(self.opt['N_sms']):
            # get the indices of neurons participating in the motif
            spike_list = []
            if self.opt['N_involved'][k]<self.opt['N_pre']:
                all_ind = torch.randperm(self.opt['N_pre'])
                self.ind_inv = all_ind[:int(self.opt['N_involved'][k])]
                for n in self.ind_inv:
                    spike_time = torch.randint(int(2*self.opt['temporal_jitter']),int(self.opt['N_timesteps']-2*self.opt['temporal_jitter']), [1]).item()
                    spike_list.append((n,spike_time))
                    self.SMs[k,n] = gaussian_kernel(self.opt['N_timesteps'],spike_time,self.opt['temporal_jitter'])
            else:
                for n in range(self.opt['N_pre']):
                    spike_time = torch.randint(int(2*self.opt['temporal_jitter']),int(self.opt['N_timesteps']-2*self.opt['temporal_jitter']), [1]).item()
                    spike_list.append((n,spike_time))
                    self.SMs[k,n] = gaussian_kernel(self.opt['N_timesteps'],spike_time,self.opt['temporal_jitter'])
            self.spike_times.append(spike_list)
    
    def draw_input(self, nb_trials=10):
        torch.manual_seed(self.opt['seed'])
        raster_plot = torch.zeros(self.opt['N_sms']+1, nb_trials, self.opt['N_pre'], self.opt['N_timesteps'], device=self.device)
        nb_spikes_noise = self.opt['additive_noise']*self.opt['proba_sms'].sum()/(1-self.opt['additive_noise'])
        random_noise = torch.poisson(nb_spikes_noise.unsqueeze(0).unsqueeze(0).repeat(nb_trials, self.opt['N_pre']))
        random_jitter = torch.normal(0, self.opt['temporal_jitter'], size=(nb_trials, self.opt['N_sms'], self.opt['N_pre']))
        random_selection = torch.bernoulli(torch.ones_like(random_jitter)*(1-self.opt['dropout_proba']))
        print(random_selection.shape)
        random_warping = (1 - self.opt['min_warping_coef'])*torch.rand(nb_trials,self.opt['N_sms']) + self.opt['min_warping_coef']
        #random_kernel = torch.distributions.Categorical(self.opt['proba_sms'].unsqueeze(0).repeat(nb_trials,1)).sample()
        random_kernel = torch.bernoulli(self.opt['proba_sms'].unsqueeze(0).repeat(nb_trials,1))
        
        for trial in range(nb_trials):
            for kernel in range(len(self.spike_times)):
                if random_kernel[trial,kernel]:
                    for spike in self.spike_times[kernel]:
                        neuron = spike[0]
                        if random_selection[trial,kernel,neuron]:
                            time = int((spike[1]-self.opt['N_timesteps']//2)*random_warping[trial,kernel]+self.opt['N_timesteps']//2+random_jitter[trial,kernel,neuron])
                            if time<self.opt['N_timesteps'] and time>=0:
                                raster_plot[kernel,trial,neuron,time] += 1
            for neuron in range(self.opt['N_pre']):
                times = torch.randint(self.opt['N_timesteps'],[int(random_noise[trial,neuron].item())])
                raster_plot[-1,trial,neuron,times] += 1
        return raster_plot, random_kernel 

def generate_dataset(parameters, record_path='../synthetic_data_no_bernoulli/', verbose=True,  device='cpu'):

    if not os.path.exists(record_path):
        os.mkdir(record_path)

    model_path = record_path+f'generative_model_'+get_dataset_parameters(parameters)
    dataset_path = record_path+f'synthetic_rp_trainset_{parameters['N_samples']}_'+get_dataset_parameters(parameters)+'.pt'
    
    if os.path.exists(model_path):
        if verbose: print(model_path)
        torch.serialization.add_safe_globals([sm_generative_model])
        sm = torch.load(model_path, map_location=device, weights_only = True)
        sm.device = device
    else:
        sm = sm_generative_model(parameters, device=device)
        torch.save(sm, model_path)

    if os.path.exists(dataset_path):
        dataset_input_list, dataset_output_list = torch.load(dataset_path, map_location=device, weights_only = True)
        dataset_input = torch.zeros(parameters['N_sms']+1, parameters['N_samples'],parameters['N_pre'],parameters['N_timesteps'], device=device)
        motifs_occurence = torch.zeros(parameters['N_samples'],parameters['N_sms'], device=device)
        dataset_input[dataset_input_list] = 1
        motifs_occurence[dataset_output_list] = 1
    else:
        dataset_input, motifs_occurence = sm.draw_input(nb_trials=parameters['N_samples'])
        dataset_input_list = torch.where(dataset_input==1)
        dataset_output_list = torch.where(motifs_occurence==1)
        torch.save((dataset_input_list, dataset_output_list), dataset_path)

    return sm, dataset_input, motifs_occurence

def kfold_dataset(dataset_input, k=3, shuffle_indices=False, existing_indices=None):

    trainsets_input, testsets_input = [],[]
    num_samples = dataset_input.shape[0]
    n_samples_per_fold = num_samples//k
    if existing_indices is not None:
        shuffled_indices = existing_indices
    elif shuffle_indices:
        shuffled_indices = torch.randperm(num_samples)
    else:
        shuffled_indices = torch.arange(num_samples)
        
    for i in range(k):
        all_indices = torch.arange(num_samples)
        testset_indices = (all_indices>=i*n_samples_per_fold)*(all_indices<(i+1)*n_samples_per_fold)
        trainsets_input.append(dataset_input[shuffled_indices[~testset_indices]])
        testsets_input.append(dataset_input[shuffled_indices[testset_indices]])
    if k==1: 
        trainsets_input = testsets_input
        print('with k=1, train and test sets are the same')
    return trainsets_input, testsets_input, shuffled_indices

def make_allen_dataset(dataset_path, number_samples_per_image=10, number_folds=5, device='cpu'):
    
    files_list = glob.glob(dataset_path)
    all_trainsets, all_testsets, all_othersets = [], [], []
    ids = torch.arange(20).repeat_interleave(10)
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

    