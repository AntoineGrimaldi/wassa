import torch

class WassA(torch.nn.Module):
    def __init__(self, kernel_size, weight_init=None, output='linear', do_bias=False, device='cpu'): 
        super(WassA, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.do_bias = do_bias
        self.output = output

        if weight_init is not None:
            if weight_init == 'flat':
                weights = torch.ones(kernel_size)
            else:
                assert list(kernel_size) == list(weight_init.size()), print(f'weights size is {weight_init.size()} when {kernel_size} is expected')
                weights = weight_init.detach().cpu().clone()
                sum_weights = torch.linalg.norm(weights, ord=2, dim=(1,2))
                weights[sum_weights==0] = torch.ones([kernel_size[1],kernel_size[2]])
        else:
            weights = torch.rand(kernel_size)

        weights.div_(torch.linalg.norm(weights, ord=2, dim=(1,2), keepdim=True)).repeat(1,weights.shape[1],weights.shape[2])
        self.decoding_weights = torch.nn.Parameter(weights.to(device), requires_grad=True)

        if do_bias:
            flat_k = torch.ones_like(weights[0]).unsqueeze(0)
            flat_k.div_(torch.norm(flat_k,p=2, dim=(1,2), keepdim=True))
            self.flat_k = flat_k.to(device)
            self.bias = torch.nn.Parameter(torch.ones([1]).to(device), requires_grad=True)

    def forward(self, input):
        
        factors = torch.nn.functional.conv1d(input, self.decoding_weights)
        if self.do_bias:
            bias_activation = self.bias*torch.nn.functional.conv1d(input, self.flat_k)
            
        if self.output == 'linear':
            output = factors
        elif self.output == 'sigmoid':
            output = torch.sigmoid(factors)
        elif self.output == 'softmax':
            output = torch.nn.functional.softmax(factors, dim=1)
            
        estimated_input = torch.nn.functional.conv_transpose1d(output, self.decoding_weights)
        if self.do_bias:
            estimated_input += torch.nn.functional.conv_transpose1d(bias_activation, self.flat_k)
        
        return output, estimated_input
    

class WassA_flex(torch.nn.Module):
    def __init__(self, kernel_size, weight_init=None, output='linear', do_bias=False, device='cpu'): 
        super(WassA_flex, self).__init__()
        self.device = device
        self.kernel_size = kernel_size
        self.do_bias = do_bias
        self.output = output

        if weight_init is not None:
            assert kernel_size == weight_init.shape, print(f'weights size is {weight_init.shape} when {kernel_size} is expected')
            weights = weight_init.detach().clone()
        else:
            weights = torch.rand(kernel_size)
        weights.div_(torch.norm(weights,p=2, dim=(1,2), keepdim=True))
        self.encoding_weights = torch.nn.Parameter(weights.to(device), requires_grad=True)
        self.decoding_weights = torch.nn.Parameter(weights.to(device), requires_grad=True)
        if do_bias:
            self.bias = torch.nn.Parameter((torch.rand(kernel_size[0])).abs().to(device), requires_grad=True)

    def forward(self, input):
        
        factors = torch.nn.functional.conv1d(input, self.encoding_weights)
        if self.do_bias:
            factors =+ torch.ones_like(factors)*self.bias[None,:,None]
            
        if self.output == 'linear':
            output = factors
        elif self.output == 'sigmoid':
            output = torch.sigmoid(factors)
        elif self.output == 'softmax':
            output = torch.nn.functional.softmax(factors, dim=1)
            
        estimated_input = torch.nn.functional.conv_transpose1d(output, self.decoding_weights)
        
        return output, estimated_input

