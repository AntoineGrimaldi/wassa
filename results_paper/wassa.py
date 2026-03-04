import torch

class WassA(torch.nn.Module):
    def __init__(self, training_parameters, device='cpu'):
        super().__init__()

        self.do_bias = training_parameters['do_bias']
        self.activation = training_parameters['activation']
        self.norm = training_parameters['kernels_norm']
        self.sigmoid = training_parameters['sigmoid']

        if training_parameters['weight_init']=='flat':
            weights = torch.ones(training_parameters['kernel_size'],device=device)
        else:
            weights = torch.rand(training_parameters['kernel_size'],device=device)
        self.weights = torch.nn.Parameter(weights, requires_grad=True)

        if self.do_bias:
            self.bias = torch.nn.Parameter(torch.zeros([training_parameters['kernel_size'][1]]).to(device), requires_grad=True)
        with torch.no_grad():
            self.normalization()

    def forward(self, x):
        
        n_sample = x.shape[0]
        n_sms, n_pre, n_timesteps = self.weights.shape
        
        if self.activation == 'emd like':
            cdf_x = torch.cumsum(x,dim=-1)
            cdf_weights = torch.cumsum(self.weights,dim=-1)
            z = torch.nn.functional.conv1d(cdf_x, cdf_weights)/n_pre
        else:
            z = torch.nn.functional.conv1d(x, self.weights)
            
        if self.sigmoid:
            z = torch.nn.functional.sigmoid(z)
            
        x_hat = torch.nn.functional.conv_transpose1d(z, self.weights)
        
        if self.do_bias:
            x_hat += self.bias.unsqueeze(0).unsqueeze(2).repeat(n_sample,1,n_timesteps)
            
        return z, x_hat

    def normalization(self):
        if self.do_bias:
            self.bias.clamp_(min=0)
        if self.norm == 1:
            self.weights.data = torch.nn.functional.normalize(self.weights.data, p=1, dim=2)
        elif self.norm == 2:
            if self.activation == 'emd like':
                cdf_weights = torch.cumsum(self.weights.data,dim=-1)
                cdf_norm = torch.nn.functional.normalize(cdf_weights, p=2, dim=(1,2))
                first_column = torch.zeros([self.weights.shape[0],self.weights.shape[1],1],device=self.weights.device)
                augmented_cum_sum = torch.cat((first_column,cdf_norm),dim=-1)
                self.weights.data = augmented_cum_sum.diff()
            elif self.activation == 'conv':
                self.weights.data = torch.nn.functional.normalize(self.weights.data, p=2, dim=(1,2))
            else:
                print('ERROR: wrong activation type')

