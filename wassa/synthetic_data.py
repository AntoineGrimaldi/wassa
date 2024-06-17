
class default_parameters():
    N_pre = 100 # number of neurons
    N_delays = 51 # number of timesteps in spiking motifs, must be a odd number for convolutions
    N_SMs = 4 # number of structured spiking motifs
    N_timesteps = N_delays*10 # number of timesteps for the raster plot (in a.u.)
    N_involved = 100 # number of neurons involved in the spiking motif
    avg_fr = 20 # average firing rate of the neurons 
    if avg_fr.shape<N_pre:
        
    
    temporal_jitter = 1 # temporal jitter for the structured motifs 
                        # (inversely proportional to the precision of spike timings)
    noise_coef = 0.1 # coefficient for the background noise/spontaneous activity
    activation_proba = 1 # probability of the neurons to be active during the generation of the spiking motif
    time_warping_coef = 1 # coefficient for time warping of the spiking motifs
    overlapping_sequences = False # condition to have overlapping patterns or sequential activation of the different kernels
    p_out_distrib = np.ones([N_SMs+1])/(N_SMs+1) # prior distribution of the output probabilities of each pattern
    seed = 666 # seed

    def get_parameters(self):
        return f'{self.N_pre}_{self.N_delays}_{self.N_SMs}_{self.N_timesteps}_{self.temporal_jitter}_{self.noise_coef}_{self.structured}_{self.seed}'


def gaussian_kernel(n_steps, mu, std):
    x = torch.arange(n_steps)
    return torch.exp(-(x-mu)**2/(2*std**2))/(std*torch.sqrt(torch.Tensor([2*torch.pi])))

class sm_generative_model_one_spike:
    def __init__(self, opt, device='cpu'):
        self.opt: Params = opt
        self.device = device
        torch.manual_seed(opt.seed)
        self.SMs = torch.zeros(self.opt.N_SMs+1, self.opt.N_pre, self.opt.N_delays).to(device)
        for k in range(self.opt.N_SMs):
            if self.opt.structured:
                angles = torch.pi/self.opt.N_SMs*(torch.arange(self.opt.N_SMs))+torch.pi/4
                for n in range(self.opt.N_pre):
                    delay, jitter = self.opt.N_delays//2 + torch.tan(angles[k])*(n-self.opt.N_pre//2), torch.normal(torch.Tensor([self.opt.temporal_jitter]), torch.Tensor([self.opt.std_temporal_jitter])).abs()
                    self.SMs[k+1, n, :] += gaussian_kernel(self.opt.N_delays, delay, jitter).to(device)
            else:
                for n in range(self.opt.N_pre):
                    delay, jitter = torch.randint(self.opt.N_delays, [1]), torch.normal(torch.Tensor([self.opt.temporal_jitter]), torch.Tensor([self.opt.std_temporal_jitter])).abs()
                    self.SMs[k+1, n, :] += gaussian_kernel(self.opt.N_delays, delay, jitter).to(device)

        if self.opt.structured:
            self.SMs[0] = self.SMs[1:].sum()/(self.opt.N_SMs*self.opt.N_pre*self.opt.N_delays)
        else:
            self.SMs[0] = 1/self.opt.N_delays

        ind_zero = self.SMs.sum(dim=-1)==0
        self.SMs.div_(torch.norm(self.SMs, p=1, dim=-1, keepdim=True)+1e-14)
        self.SMs[ind_zero] = 1/self.opt.N_delays

    def draw_input(self, nb_trials=1):
        input_rp = torch.zeros(nb_trials, self.opt.N_pre, self.opt.N_timesteps, device=self.device)
        output_rp = torch.zeros(nb_trials, self.opt.N_SMs+1, self.opt.N_timesteps, device=self.device)

        nb_motifs = self.opt.N_timesteps//self.opt.N_delays
        times = torch.arange(1,nb_motifs+1)*(self.opt.N_delays)
        times = times.repeat(nb_trials,1)
            
        k_sms = torch.tensor(np.random.choice(np.arange(self.opt.N_SMs+1), size=(nb_trials,nb_motifs), p=self.opt.p_out_distrib))

        for tr in range(nb_trials):
            for mo in range(nb_motifs):
                k_sm, time = k_sms[tr,mo], times[tr,mo]
                input_rp[tr,:,time-self.opt.N_delays:time] += torch.bernoulli((self.opt.noise_coef*self.SMs[0]+(1-self.opt.noise_coef)*self.SMs[k_sm]).squeeze(0))
                output_rp[tr,k_sm, time-1] = 1
            
        return input_rp, output_rp