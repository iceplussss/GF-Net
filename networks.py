'''The class of the generative Fermonic neural network states
'''

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import logging
from network_modules import *
from util import *
 
class GFNet(nn.Module):
    """The generative fermionic neural network state. 
    """
    def __init__(self, ncell, Ecell, charge, groups=1, nocpd = None,
                amp_hidden=32, phase_hidden = 32,
                dilation_depth=3, kernel_size=2,
                ):
        super(GFNet, self).__init__()
        ####################### Physical model information ######################
        self.ncell = ncell  ## number of single-electron orbital
        self.cellsize = 2  ## 2**cellsize is the degree of freedom of orbital occupation/ tensor index
        self.charge = charge  ## total number of electrons
        self.groups = groups  ## number of parallel training
        ## sz conservation
        self.nocpd = nocpd

        ####################### Spectral properties   ######################
        self.amp = nn.Parameter(th.zeros(self.groups)-3)
        self.register_buffer('expH', th.zeros(self.groups))

        ####################### Network Hyper-parameters ######################
        ## amp-net parameters
        self.amp_hidden = amp_hidden        ## amp_net hidden channel
        self.phase_hidden = phase_hidden    ## phase_net hidden channel
        self.n_output = 2**self.cellsize      ## dof of each tensor index
        self.kernel_size = kernel_size  
        self.dilation_depth = int(np.ceil(np.log(self.ncell-1) / np.log(2)))
        # self.dilation_depth = dilation_depth
        self.dilations = [2 ** i for i in range(self.dilation_depth)]
        self.receptive_field = 2 + (self.kernel_size - 1) * sum(self.dilations) + 1
        assert self.receptive_field >= self.ncell
        ## Define neural network
        self.amp_net = amp_net_mixed(ncell=self.ncell, Ecell=Ecell, dilations=self.dilations, 
            cellsize=self.cellsize, groups=groups, 
            hd_ch=amp_hidden, n_output=self.n_output, kernel_size=kernel_size,
            )
    @property
    def amplitude(self):
        return th.exp(self.amp)

    def data_parallel(self):
        ''' put two neural networks into DataParallel mode, second dimension is distributed'''
        self.amp_net = nn.DataParallel(self.amp_net, dim=1)

    def _reassign(self, x):
        return x.to(device=self.expH.device)

    def _select(self, bin_confs, x):
        '''collect output according to input spin configuration
        Args:
            inputs (Tensor): tensor with the shape (batch, ncell).
            x(Tensor) : unnormalized wave-function with shape (batch, ncell, 2^cellsize)
        Returns:
            x_selected(Tensor): tensor with shape (batch,  ncell)
        '''
        select_idx = bit2integer(bin_confs.type(th.long),2)
        return th.gather(x,-1,select_idx.unsqueeze(-1)).squeeze(-1)
    
    def _preprocess(self,bin_basis, for_phase=False):
        '''
        For amp_net: Add auxiliary orbital on top, take away physical orbital at bottom. 
        Args:
            configuration (Tensor or np.array): tensor with the shape (batch, norbs,2).
        Returns:
            output(th.tensor, float type): tailored configuration, shape=(1,batch, cellsize, ncell).  
        '''
        output = bin_basis.unsqueeze(0).transpose(-1,-2).type_as(self.expH) # (1,nbatch,cellsize,ncell)
        if for_phase == True:  ## pbc can be directly handled here without worrying generative function        
            return output
        else:
            extra = th.zeros_like(output[:,:,:,:1])
            output = th.cat([extra, output[:,:,:,:-1]],-1)
            return output

    def _CSmask(self, s, cell_occp):
        '''get mask for enforcing charge conservation at one spin direction.
        Args:
            inputs (Tensor,th.long): tensor with the shape (1, batch, ncell).
            cell_occp(Tensor, th.long): the occupation of each possible cell configuration, (2**cellsize)
        Returns:
            mute_mask(Tensor, th.bool): (1, batch, ncell,2^cellsize)
        '''
        s_occp_cumsum = s.cumsum(-1)  #(1, nbatch, ncell)
        s_empt = (1-s)
        s_empt[:,:,0] *= 0  ## silence the auxiliary orbital
        s_empt_cumsum = s_empt.cumsum(-1)  #(1, nbatch, ncell)
        mute_occp = (s_occp_cumsum[:,:,:,None] + cell_occp[None,None,None,:]) > self.nocpd
        mute_empt = (s_empt_cumsum[:,:,:,None] + (1-cell_occp)[None,None,None,:]) > s.shape[-1]-self.nocpd
        mute_mask = (mute_occp.cumsum(-2)>0) | (mute_empt.cumsum(-2)>0)
        return mute_mask

    def _postprocess(self, inputs, amplitude):
        '''exert normalization and charge/spin conservation
        Args:
            inputs (Tensor or np.array): tensor with the shape (1, batch, cellsize, ncell).
            amplitude(Tensor) : unnormalized wave-function with shape (1, batch, ncell, 2^cellsize)
        Returns:
            amplitude(Tensor): normalized amplitude, tensor with shape (batch, ncell, 2^cellsize)
        '''
        s = inputs.transpose(-1,-2)  #(1, batch, ncell,cellsize)
        groups, nbatch, ncell, cellsize = s.shape
        cell_occp = integer2bit(
                th.arange(2**cellsize),cellsize
                ).to(dtype=s.dtype, device=s.device)
        mute_mask_0 = self._CSmask(s[:,:,:,0],cell_occp[:,0])
        mute_mask_1 = self._CSmask(s[:,:,:,1],cell_occp[:,1])
        mute_mask = mute_mask_0 | mute_mask_1
        alive_mask = ~mute_mask
        amplitude = amplitude * alive_mask.type_as(amplitude)

        ############################## sanity check ######################################
        amp2 = (amplitude.detach()**2).sum(-1)   #(groups, nbatch, ncell)
        anm_flag = (amp2 <= 0)
        if amp2.min()<= 0:
            anomaly = (anm_flag.type(th.long).sum(-1)) > 0  
            logging.debug("{} anomaly detected and patched".format(anomaly.type(th.long).sum()))
            ## patch the anomaly
            needs_regular = anm_flag.to(dtype=th.long)
            amplitude += (alive_mask.to(dtype=th.long) * needs_regular.unsqueeze(-1)).type_as(amplitude)
            
        ##################################################################################
        ## re-normalize probability distribution
        amplitude = amplitude / ((amplitude**2).sum(-1).unsqueeze(-1))**0.5   ## normalized amplitude
        return amplitude[0]

    def forward(self, bin_confs):
        """FORWARD CALCULATION.
        Args:
            bin_confs (th.long): binary configurations, 
                tensor with the shape (batch, norbs, 2).
        Returns:
            psi_r(i) (Tensor): Real(imaginary) part of the wave function with the shape (batch) .
        """
        preped_config = self._preprocess(bin_confs)
        amp = self.amp_net(preped_config) 
        amp = self._postprocess(preped_config, amp)
        amp = self._select(bin_confs, amp)
        amp = amp.prod(-1)
        psi_r = amp
        psi_i = th.zeros_like(amp)
        return psi_r, psi_i

    def forward_unique(self, bin_confs):
        unique_input, inverse_idx, counts = th.unique(bin_confs, return_inverse=True,return_counts=True, dim=0)
        unique_output_r, unique_output_i = self.forward(unique_input)
        output_r = th.gather(unique_output_r,dim=0,index = inverse_idx)
        output_i = th.gather(unique_output_i,dim=0,index = inverse_idx)
        return output_r, output_i

    def generate(self, nbatch, gamma=0.8):
        """GENERATE configuration with respect to each group according to wave amplitude |psi|^2
        Args:
            nbatch: size of the batch going to be generated 
        Returns:
            samples(Tensor): Generated Configuration with shape (groups, nbatch, in_dim).
            samples_weight(Tensor): amplitude^2 of the generated configuration with shape (groups, nbatch)
        """
        bin_samples = th.zeros((nbatch, self.ncell,self.cellsize),
                    device=self.expH.device, dtype = th.long )   ## to be sampled cell-wisely
        with th.no_grad():
            for current_cell in range(self.ncell):
                preped_config = self._preprocess(bin_samples)
                amp_output = self.amp_net(preped_config)
                amp_adjusted = th.abs(amp_output)**gamma
                amp_normed = self._postprocess(preped_config, amp_adjusted) ## (nbatch,ncell, 2^cellsize)
                prob_current = amp_normed[:,current_cell, :]**2 ## (nbatch, 2^cellsize)
                #########
                selected_idx = th.multinomial(prob_current,1).reshape(nbatch)
                selected_prob = th.gather(
                    prob_current, -1, selected_idx.unsqueeze(-1)
                                            ).squeeze(-1)
                bin_samples[:,current_cell] = integer2bit(selected_idx, self.cellsize)
                if current_cell == 0:
                    prob_final =  selected_prob
                else:
                    prob_final *= selected_prob                     
        return bin_samples, prob_final
    