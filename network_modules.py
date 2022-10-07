''' Subclass of the PyTorch module. Components of GFNet.
'''
import math
import torch as th
from torch import nn
from torch.nn import Module, Parameter, init, Sequential
from torch.nn import Conv1d, Conv2d, Linear

class SimulatedComplexLinear(Module):
    '''
    input : (groups, batch, in_features)
    output: (groups, batch, out_features)
    '''
    __constants__ = ['bias', 'in_features', 'out_features', 'groups']
    def __init__(self, in_features, out_features, groups = 1, bias=True):
        super(SimulatedComplexLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.bias = bias
        self.weight = Parameter(th.Tensor(groups, out_features, in_features))
        if bias:
            self.bias = Parameter(th.Tensor(groups, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError("dim(input) has to be 3") 
        if x.shape[0] != self.groups or x.shape[2] != self.in_features:
            raise ValueError("Input shape must be (groups, batch, in_features)") 
        output = th.bmm(x, self.weight.transpose(1,2))
        if self.bias is not None:
            output += self.bias[:,None,:]
        return output
        # output_r = output[:,:,:self.out_features]
        # output_i = output[:,:,self.out_features:]
        # return output_r, output_i

    def extra_repr(self):
        return 'in_features={}, out_features={}, groups={}, bias={}'.format(
            self.in_features, self.out_features, self.groups, self.bias is not None
        )

class CausalConv1d(Module):
    """1D DILATED CAUSAL CONVOLUTION."""
    def __init__(self, in_channels, out_channels, 
                 kernel_size, dilation=1, groups=1, bias=True):
        super(CausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = 1
        self.dilation = dilation
        self.groups = groups
        self.padding = padding = (kernel_size - 1) * dilation
        self.conv = Conv1d(in_channels*groups, out_channels*groups, kernel_size,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (groups, batch, in_channel, height).
        Returns:
            Tensor: Float tensor variable with the shape (groups, batch, out_channel, height)
        """
        if x.dim() != 4:
            raise NotImplementedError("The total dimension of input should be 4, but get {}".format(x.dim()))
        groups, nbatch, in_ch, in_height = x.shape
        if groups == self.groups:
            x = x.transpose(0,1).reshape(nbatch, groups*in_ch,in_height)
        else:
            raise ValueError("input shape is illegal")
        x = self.conv(x)
        x = x.reshape(nbatch,groups,self.out_channels,-1).transpose(0,1)
        if self.padding != 0:
            x = x[:, :, :, :-self.padding]
        assert x.shape[-1] == in_height
        return x

class NonCausalConv1d(Module):
    """1D DILATED CAUSAL CONVOLUTION."""
    def __init__(self, in_channels, out_channels, 
                dilation=1, groups=1, bias=True):
        super(NonCausalConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 3
        self.stride = 1
        self.dilation = dilation
        self.groups = groups
        self.conv = Conv1d(in_channels*groups, out_channels*groups, self.kernel_size,
                              padding=self.dilation, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        """FORWARD CALCULATION.
        Args:
            x (Tensor): Float tensor variable with the shape  (groups, batch, in_channel, height).
        Returns:
            Tensor: Float tensor variable with the shape (groups, batch, out_channel, height)
        """
        if x.dim() != 4:
            raise NotImplementedError("The total dimension of input should be 4, but get {}".format(x.dim()))
        groups, nbatch, in_ch, in_height = x.shape
        if groups == self.groups:
            x = x.transpose(0,1).reshape(nbatch, groups*in_ch,in_height)
        else:
            raise ValueError("input shape is illegal")
        x = self.conv(x)
        x = x.reshape(nbatch,groups,self.out_channels,in_height).transpose(0,1)
        return x

class amp_net_mixed(Module):
    """Component of the GFNet
    """
    def __init__(self, ncell, Ecell, dilations, cellsize=2, groups=1, 
                    hd_ch=32, n_output=4, kernel_size=2):
        super(amp_net_mixed, self).__init__()
        self.ncell = ncell
        self.cellsize = cellsize
        self.groups = 1
        self.hd_ch = hd_ch        ## convolution channel 
        self.n_output = n_output
        self.kernel_size = kernel_size
        self.dilations = dilations
        ## first layer
        self.causal_in = CausalConv1d(in_channels=cellsize, out_channels=hd_ch, 
                 kernel_size=kernel_size, dilation=1, groups=groups, bias=True)
        ## convolution layers
        self.dil_act = nn.ModuleList()

        for d in self.dilations:
            self.dil_act += [CausalConv1d(hd_ch, hd_ch, kernel_size,
                                                    dilation=d, groups=groups, bias=True)]
        self.skip_conv = CausalConv1d(hd_ch*len(self.dilations), hd_ch, kernel_size=1,
                                                dilation=1, bias = True )
        self.local_linear = SimulatedComplexLinear(hd_ch, hd_ch, groups = ncell, bias = True)
        self.final_linear = SimulatedComplexLinear(hd_ch, n_output, groups = ncell, bias = True)
        self.cellocp = th.tensor([[0,1,1,2]],dtype=Ecell.dtype,device=Ecell.device)
        # self.register_buffer('cell_weight', th.exp(- Ecell.unsqueeze(-1) * ft))
        self.Ecell = nn.Parameter(Ecell)
    # @profile_every(1)
    def seq_forward(self, x):
        """FORWARD CALCULATION - amplitude part.
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch, n_skipth, ncell) .
        """
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x)
        x = self.skip_2x2(x) / self.n_resch  #(1, batch, self.ngroup * nskipth, ncell)

        x = x.permute(0,-1,1,2).reshape(self.ncell,-1, self.ngroup * self.n_skipch)
        x = self.local_linear(x)
        x = x.reshape(self.ncell, -1, self.ngroup, self.n_skipch).permute(2,1,3,0)
        return x
    def skipadd_forward(self, x):
        """FORWARD CALCULATION - amplitude part.
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch, n_skipth, ncell) .
        """
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        skip = 0
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x)
            skip += self.skip_1x1[i](x)  / self.n_resch #(1, batch, self.ngroup * nskipth, ncell)
        skip = lrelu(skip)
        skip = skip.permute(0,-1,1,2).reshape(self.ncell,-1, self.ngroup * self.n_skipch)
        skip = self.local_linear(skip)
        skip = skip.reshape(self.ncell, -1, self.ngroup, self.n_skipch).permute(2,1,3,0)
        return skip
    def forward(self, x):
        """FORWARD CALCULATION - amplitude part.
        Args:
            configuration (Tensor): tensor with the shape (1,batch, channel, ncell).
            the configuration is already reordered if pbc is true
        Returns:
            output(Tensor): wave function amplitude with the shape (groups, batch,  ncell, n_output) .
        """
        lrelu = nn.LeakyReLU(0.1)
        tanhsh = nn.Tanhshrink()
        x = self.causal_in(x)
        x = tanhsh(x)
        skip = []
        for i, dil_ly in enumerate(self.dil_act):
            x = dil_ly(x)
            x = lrelu(x) # #(1, batch, hd_ch, ncell)
            skip.append(x.clone())
        skip = th.cat(skip, 2) #(1, batch, hd_ch*layers, ncell)
        skip = self.skip_conv(skip) #(1, batch, hd_ch, ncell)
        skip= lrelu(skip)
        skip = skip.squeeze(0).permute(-1,0,1) #(ncell,batch, hd_ch)
        skip = self.local_linear(skip) #(ncell,batch, hd_ch)
        skip = lrelu(skip)
        skip = self.final_linear(skip) # (ncell,batch, n_output)
        skip = skip.permute(1,0,2).unsqueeze(0)
        # skip = skip * self.cell_weight[None,None,:,:]
        skip = skip * th.exp(- self.Ecell.unsqueeze(-1) * self.cellocp)
        return skip
