"""Contains simple helper functions and classes
"""
import torch as th
import numpy as np
from math import factorial
from collections import namedtuple
from pyscf import ao2mo 

def th2np(x, digits = None):
    np_x = x.cpu().detach().numpy()
    result = np_x if digits == None else np.round(np_x, digits)
    return result

def parse_xyz(filename, basis='ccpvdz', verbose=0):
    with open(filename) as fp:
        natoms = int(fp.readline())
        comments = fp.readline()
        xyz_str = "".join(fp.readlines())
    mol = gto.Mole()
    mol.verbose = verbose
    mol.atom = xyz_str
    mol.basis  = basis
    mol.build(0,0,unit="Ang")
    return mol  

def integer2bit(integer, num_bits=8):
    """Turn integer tensor to binary representation.
        Args:
            integer : torch.Tensor, tensor with integers
            num_bits : Number of bits to specify the precision. Default: 8.
        Returns:
            Tensor: Binary tensor. Adds last dimension to original tensor for
            bits.
    """
    result = [((integer.unsqueeze(-1)) >> (num_bits-1-i)
                        )%2 for i in range(num_bits)
                        ]
    return th.cat(result,-1)

def bit2integer(bits, num_bits):
    exponent_mask =  num_bits - 1 - th.arange(num_bits,dtype=bits.dtype, device=bits.device).expand_as(bits)
    exponent_mask = 2**exponent_mask
    select_idx = (bits * exponent_mask).sum(-1)
    return select_idx.type(th.long)

def lr_scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def dict2obj(d):
    return namedtuple("config", d.keys())(*d.values())

def parity(x):
    nbatch, ndim = x.shape
    parity = 0
    for i in range(1,ndim):
        parity += (x[:,i].unsqueeze(-1) < x[:,:i]).type(th.long).sum(-1)
    parity = (-1)**(parity%2)
    return parity
 

def get_h1e(mol, orb):
    # return <i|h1|j> where i,j are molecule orbitals
    h_ao = mol.intor_symmetric('int1e_kin') + mol.intor_symmetric('int1e_nuc')
    # orb is [nao x norbital], each column stands for an orbital
    h_mo = np.einsum("ip,pq,qj->ij", orb.conj().T, h_ao, orb)
    return h_mo

def get_eri(mol, orb, compact=False, phy_order=True):
    # chemistry notation eri[i,j,k,l] = (ij|kl)   = \int \phi_i*(r_1) \phi_j(r_1) 1/|r_1-r_2| \phi_k*(r_2) \phi_l(r_2)
    # physics notation   eri[i,j,k,l] = <ij|g|kl> = \int \phi_i*(r_1) \phi_j*(r_2) 1/|r_1-r_2| \phi_k(r_1) \phi_l(r_2)
    # (ij|kl) = <ik|g|jl>
    eri = ao2mo.full(mol, orb, compact=compact)
    if compact:
        return eri # chem notation
    else:
        eri = eri.reshape(mol.nao, mol.nao, mol.nao, mol.nao)
        if phy_order:
            eri = eri.transpose((0,2,1,3))
        return eri


class mc_sampler:
    """ Parallel Monte Carlo sampling
    """
    def __init__(self, net_ensemble):
        self.net_ensemble = net_ensemble
        self.sample = None
        self.weight = None
        self.counter = -1
        self.gamma = 0.5
    def get_samples(self, nbatch):
        '''
        sample (2*nbatch) samples for recycling. Take out [:nbatch], [0.5nbatch,1.5nbatch],[nbatch:],[1.5batch:0.5nbatch]
        for 4 gradient descent
        Returns:
            sampled spin configuration of shape=(nbatch,nspin)
            associated weight of shape (nbatch)
        '''
        recycle=4
        extend_size = 2
        self.counter = (self.counter + 1) % recycle
        if self.counter % recycle == 0:
            self.sample, self.weight = self.net_ensemble.generate(nbatch*extend_size, self.gamma)
        half_batch = nbatch // 2
        lhalf_idx = th.arange(half_batch) + self.counter * half_batch
        rhalf_idx = th.arange(half_batch) + (self.counter+1)%recycle * half_batch
        output_sample = self.sample[th.cat([lhalf_idx,rhalf_idx])]
        output_weight = self.weight[th.cat([lhalf_idx,rhalf_idx])]
        return output_sample, output_weight  