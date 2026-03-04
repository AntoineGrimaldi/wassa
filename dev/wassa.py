import torch

class WassA(torch.nn.Module):
    def __init__(self, training_parameters, device='cpu'):
        super().__init__()

        K, N, D = training_parameters['kernel_size']
        self.tied_weights = training_parameters['tied_weights']
        self.do_bias = training_parameters['do_bias']
        self.top_k = training_parameters['topk']
        self.norm = training_parameters['kernels_norm']
        
        if self.tied_weights:
            if training_parameters['weight_init']=='flat':
                weights = torch.ones(training_parameters['kernel_size'],device=device)
            else:
                weights = torch.rand(training_parameters['kernel_size'],device=device)
            self.weights = torch.nn.Parameter(weights, requires_grad=True)
        else:
            self.encoder = torch.nn.Sequential(
                    torch.nn.Conv1d(N, K, kernel_size=(D), bias=self.do_bias[1],device=device),
                    torch.nn.Sigmoid()
            )
            self.decoder = torch.nn.Sequential(
                    torch.nn.ConvTranspose1d(K, N, kernel_size=(D), bias=self.do_bias[2],device=device),
            )
        
            if training_parameters['weight_init']=='flat':
                torch.nn.init.zeros_(self.encoder[0].weight)
                self.decoder[0].weight.data.fill_(1/D)

        if self.do_bias[0]:
            self.bias = torch.nn.Parameter(torch.zeros([training_parameters['kernel_size'][1]]).to(device), requires_grad=True)
        with torch.no_grad():
            self.normalization()

    def forward(self, x):

        if self.tied_weights:
            z = torch.nn.functional.conv1d(x, self.weights)
        else:
            z = self.encoder(x)
            
        if self.top_k is not None:
            values, indices_time = z.topk(self.top_k)
            indices_trial = torch.arange(indices_time.shape[0]).unsqueeze(-1).unsqueeze(-1).repeat(1,indices_time.shape[1],indices_time.shape[2])
            indices_neurons = torch.arange(indices_time.shape[1]).unsqueeze(0).unsqueeze(-1).repeat(indices_time.shape[0],1,indices_time.shape[2])
            topz = torch.zeros_like(z)
            topz[indices_trial.flatten(),indices_neurons.flatten(),indices_time.flatten()] = values.flatten()
        else:
            topz = z
            indices_time = None
            
        if self.tied_weights:    
            x_hat = torch.nn.functional.conv_transpose1d(topz, self.weights)
        else:
            x_hat = self.decoder(topz)

        if self.do_bias[0]:
            x_hat += self.bias.unsqueeze(0).unsqueeze(2).repeat(x.shape[0],1,x.shape[2])
        return topz, x_hat, indices_time

    def normalization(self):
        if self.do_bias[0]:
            self.bias.clamp_(min=0)
        if self.tied_weights:
            if self.norm == 1:
                self.weights.data = torch.nn.functional.normalize(self.weights.data, p=1, dim=2)
            elif self.norm == 2:
                self.weights.data = torch.nn.functional.normalize(self.weights.data, p=2, dim=(1,2))
        else:
            if self.do_bias[2]:
                self.decoder[0].bias.clamp_(min=0)
            if self.norm == 1:
                self.encoder[0].weight.data = torch.nn.functional.normalize(self.encoder[0].weight.data, p=1, dim=2)
            elif self.norm == 2:
                self.encoder[0].weight.data = torch.nn.functional.normalize(self.encoder[0].weight.data, p=2, dim=(1,2))

