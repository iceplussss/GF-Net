'''The Class implementing the Variational Monte Carlo algorithm with parallel sampling
'''

import logging
import os
import numpy as np
import torch as th
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from networks import GFNet
from hamiltonian import hamiltonian
from util import *


class VMC:
    """Main simulation object. 
    Variational Monte Carlo methods for solving the ground state of the given Hamiltonian, with GFNet as the ansatz.
    Attributes:
        ham: The Hamiltonian of the electronic system.
        net: GFNet.
        device: The device for running the neural network.
        sampler: The Monte Carlo sampler
    """
    def __init__(self, ham, net_paras):
        self.ham = ham.to(device=net_paras.device, dtype=net_paras.dtype)
        self.net = GFNet(
                ncell = ham.norbs, 
                Ecell = ham.get_Eoffset(),
                charge = ham.charge, 
                nocpd = ham.nocpd,
                amp_hidden = net_paras.amp_hidden, 
                phase_hidden = net_paras.phase_hidden,
                # dilation_depth = net_paras.dilation_depth, 
                )
        self.net = self.net.to(device=net_paras.device, dtype=net_paras.dtype)
        self.device = net_paras.device
        try:
            if th.cuda.device_count() > 1:
                self.net.data_parallel()
        except:
            pass
        self.sampler = mc_sampler(self.net)
        
    def Hnet(self, net_ensemble, source_confs, evaluate_source=False, epsilon=0.001): 
        '''
        Args: 
            net_ensemble(module): the neural network used to evaluate
            source_confs(tensor): shape=(source_batch, norbs,2)
        Returns: 
            Hsita_Re, Hsita_Im   = (source_batch)
        '''            
        source_batch, N, _ = source_confs.shape
        ## conf_nbs: (n_nbs, norbs,2), hamp_nbs:(n_nbs),idx_nbss:(n_nbs)
        conf_nbs, hamp_nbs, idx_nbs = self.ham.get_all_neighbors(source_confs, epsilon)
        sita_nbs_Re, sita_nbs_Im = net_ensemble.forward_unique(conf_nbs)
        # sita_nbs_Re, sita_nbs_Im = self.cheapnet(net_ensemble, basis_nbhood.reshape(groups, -1, nspin))
        placeholder = th.zeros(source_batch,device=conf_nbs.device).type_as(sita_nbs_Re)
        Hsita_Re = th.index_add(placeholder, 0, idx_nbs, sita_nbs_Re * hamp_nbs)
        Hsita_Im = th.index_add(placeholder, 0, idx_nbs, sita_nbs_Im * hamp_nbs)
        if evaluate_source:
            return Hsita_Re, Hsita_Im, sita_nbs_Re[:source_batch], sita_nbs_Im[:source_batch]
        else:
            return Hsita_Re, Hsita_Im 
    
    def sample_variance(self, net_ensemble, E_target, epsilon=0.001, exact = True, batchsize=1000):
        '''
        This function assumes the final wave function is the unnormalized |HF>+|psi>
        So the energy expectation is <HF+psi|H|HF+psi> / <HF+psi|HF+psi>
        the variance deviation is <HF+psi|(H-E)^2|HF+psi> / <HF+psi|HF+psi>
        '''
        ham = self.ham
        amp = net_ensemble.amplitude.sum()
        hf_conf = ham.hf_base_binary()[0].unsqueeze(0) ## Hartree-Fock reference state configuration
        
        ## Get <HF|(H-E)^2|HF> and <HF|(H-E)^2|psi> exactly
        ## in the following, using Q to denote the operator(H-E_target)
        conf_hf, hamp_hf, idx_hf = self.ham.get_all_neighbors(hf_conf, epsilon)
        Qref = hamp_hf.clone()
        Qref[0] -= E_target ## (H-E)|HF>
        refQQref = (Qref**2).sum() ## <HF|(H-E)^2|HF>
        ## Hpsi at hf_conf's neighborhood
        hf_Hpsi_r, hf_Hpsi_i, hf_psi_r, hf_psi_i = self.Hnet(net_ensemble, conf_hf,evaluate_source=True, epsilon=epsilon)
        hf_Qpsi_r = (hf_Hpsi_r - hf_psi_r * E_target) * amp # Re(H-E)psi at hf_conf's neighborhood
        refQQpsi_r = (Qref*hf_Qpsi_r).sum()  ## <HF|(H-E)^2|psi>
        ## Get <psi|(H-E)^2|psi> exactly or MCly
        if exact:
            full_confs = ham.full_confs
            Hpsi_r, Hpsi_i, psi_r,psi_i = self.Hnet(net_ensemble, full_confs,evaluate_source=True, epsilon=epsilon)
            Qpsi_r = (Hpsi_r - psi_r * E_target)
            Qpsi_i = (Hpsi_i - psi_i * E_target)
            psiQQpsi = (Qpsi_r**2 + Qpsi_i**2).sum() * amp**2
            with th.no_grad():
                psiHpsi = (psi_r*Hpsi_r+psi_i*Hpsi_i).sum() * amp**2
        else:
            sampled_confs, mc_weight = net_ensemble.generate(batchsize)
            Hpsi_r, Hpsi_i, psi_r, psi_i = self.Hnet(net_ensemble, sampled_confs,evaluate_source=True, epsilon=epsilon)
            psipsi = psi_r**2 + psi_i**2
            psiQQpsi = (Hpsi_r - psi_r * E_target)**2 + (Hpsi_i - psi_i * E_target)**2
            psiQQpsi = (psiQQpsi/mc_weight).mean() / (psipsi/mc_weight).mean() * amp**2
            with th.no_grad():
                psiHpsi = psi_r*Hpsi_r + psi_i*Hpsi_i
                psiHpsi = (psiHpsi/mc_weight).mean() / (psipsi/mc_weight).mean() * amp**2
        ## get <HF+psi|(H-E)^2|HF+psi> / <HF+psi|HF+psi>
        var_nominator = refQQref + 2*refQQpsi_r + psiQQpsi
        var_denominator = 1.0 +  2*hf_psi_r[0]*amp + amp**2  ## <HF|HF> + 2Re<HF|psi> + <psi|psi>
        varH = var_nominator / var_denominator
        with th.no_grad():
            refHref = hamp_hf[0]
            refHpsi = hf_Hpsi_r[0] * amp
            expH = (refHref + 2*refHpsi + psiHpsi) / var_denominator
        return expH, varH
    
    def pretrain_loss(self, net_ensemble, epsilon=0.001):
        ##
        # |HF> + |psi> = (1-(H-E_l)dt)|HF>
        ##
        amp = net_ensemble.amplitude.sum()
        hf_conf = self.ham.hf_base_binary()[0].unsqueeze(0) ## Hartree-Fock reference state configuration
        ## in the following, using Q to denote the operator(H-E_target)
        nbs_hf_conf, hamp_hf, idx_hf = self.ham.get_all_neighbors(hf_conf, epsilon)
        dt = 0.05 ## adaptive dt
        hf_psi_r, hf_psi_i = net_ensemble(nbs_hf_conf)
        hf_Href_r = hamp_hf.clone()
        hf_Href_r[0] = 0.1
        hf_Href_i = th.zeros_like(hf_Href_r)
        # loss = ((amp*hf_psi_r + hf_Href_r*dt)**2+(amp*hf_psi_i + hf_Href_i*dt)**2).sum()
        loss = ((amp*hf_psi_r + hf_Href_r*dt)**2+(amp*hf_psi_i + hf_Href_i*dt)**2).sum()
        loss += (1 - (hf_psi_r**2+hf_psi_i**2).sum()) * amp**2
        return loss
    
    def sample_energy(self, net_ensemble,  epsilon=0.001, exact = True, batchsize=1000):
        '''
        This function assumes the final wave function is the unnormalized |HF>+|psi>
        So the energy expectation is <HF+psi|H|HF+psi> / <HF+psi|HF+psi>
        the variance deviation is <HF+psi|(H-E)^2|HF+psi> / <HF+psi|HF+psi>
        '''
        ham = self.ham
        hf_conf = ham.hf_base_binary()[0].unsqueeze(0) ## Hartree-Fock reference state configuration
        amp = net_ensemble.amplitude.sum()
        
        ## Get <HF|(H-E)^2|HF> and <HF|(H-E)^2|psi> exactly
        nbs_hf_conf, hamp_hf, idx_hf = self.ham.get_all_neighbors(hf_conf, epsilon)
        hf_Hpsi_r, hf_Hpsi_i, hf_psi_r, hf_psi_i = self.Hnet(net_ensemble, hf_conf,evaluate_source=True, epsilon=epsilon)
        hf_amp = 1 - hf_psi_r[0]
        ## Get <psi|(H-E)^2|psi> exactly or MCly
        if exact:
            full_confs = ham.full_confs
            Hpsi_r, Hpsi_i, psi_r,psi_i = self.Hnet(net_ensemble, full_confs,evaluate_source=True, epsilon=epsilon)
            psiHpsi = (psi_r*Hpsi_r+psi_i*Hpsi_i).sum() * amp**2
        else:
            # sampled_confs, mc_weight = net_ensemble.generate(batchsize)
            sampled_confs, mc_weight = self.sampler.get_samples(batchsize)
            Hpsi_r, Hpsi_i, psi_r, psi_i = self.Hnet(net_ensemble, sampled_confs,evaluate_source=True, epsilon=epsilon)
            psipsi = psi_r**2 + psi_i**2
            psiHpsi = psi_r*Hpsi_r + psi_i*Hpsi_i
            psiHpsi = (psiHpsi/mc_weight).mean() / (psipsi/mc_weight).mean() * amp**2
        refHref = hamp_hf[0] * hf_amp**2
        refHpsi = hf_Hpsi_r[0] * amp * hf_amp
        expH = (refHref + 2*refHpsi + psiHpsi) / (hf_amp**2 +  2*hf_psi_r[0]*amp*hf_amp + amp**2)
        return expH, hf_psi_r[0]

    def train(self, config, exact=False):
        '''
        Args:
            config: optim_paras
            exact: determines using Exact sampling or MC samplig
        '''
        writer = SummaryWriter(comment=config.comment)
        net = self.net
        E_lb = self.ham.E_exact   # An estimation of the lower bound of the ground energy
        E_history = []
        # E_lb = self.ham.exact_diag()
        batchsize = config.batchsize
        lr = config.lr_begin
        optimizer = th.optim.Adam([
            {'params': net.amp_net.parameters()},
            # {'params': net.phase_net.parameters()},
            {'params': net.amp, 'lr': 0.1*lr},
            ], lr = lr, betas=(0.9, 0.999))
        # optimizer = th.optim.Adam(net.parameters(), lr = lr, betas=(0.95,0.99))
        for epoch in range(config.nepoch):
            E_target = E_lb - self.ham.ccs_0 - 2
            if epoch < 0:
                if epoch % config.freq_display == 0:  
                    with th.no_grad():
                        if exact:
                            expH, hf_psi_r = self.sample_energy(net, epsilon=config.epsilon)
                        else:
                            expH, hf_psi_r = self.sample_energy(net, epsilon=config.epsilon, 
                                exact = False, batchsize=config.batchsize)
                    varH = th.tensor([0])
                loss = self.pretrain_loss(net)
                writer.add_scalar('pretrain_loss', loss, epoch)
            else:
                if exact:
                    expH,hf_psi_r = self.sample_energy(net, epsilon=config.epsilon)
                    # expH, varH = self.sample_variance(net, E_target, epsilon=config.epsilon)
                else:
                    expH,hf_psi_r = self.sample_energy(net , epsilon=config.epsilon, 
                        exact = False, batchsize=config.batchsize)
                varH = th.tensor([0])
                    # expH, varH = self.sample_variance(net, E_target, epsilon=config.epsilon, 
                    #     exact = False, batchsize=config.batchsize)
                loss = expH 
            E_pred = th2np(expH) + self.ham.ccs_0
            # if (E_pred-E_lb)<0.005:
            #     self.sampler.gamma=0.8
            E_history.append(E_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % config.freq_display == 0 :
                logging.info('================= Epoch = {} ================='.format(epoch))
                logging.info('E-E_ccsdt= {} a.u.'.format(E_pred-E_lb))
                writer.add_scalar('var_extm', varH, epoch)
                writer.add_scalar('E-E_ccsdt', E_pred-E_lb, epoch)
                writer.add_scalar('net_amp', net.amplitude, epoch)
                writer.add_scalar('hf_psi_r', hf_psi_r, epoch)
                writer.add_scalar('learning_rate', lr, epoch)
                writer.add_scalar('batchsize', batchsize, epoch)
                with th.no_grad():
                    if self.ham.nocpd < 4:
                        exact_expH, hf_psi_r = self.sample_energy(net, epsilon=config.epsilon)
                        net.expH[0] = exact_expH + self.ham.ccs_0
                    else:
                        net.expH[0] = 0.99*net.expH[0]+0.01*(expH + self.ham.ccs_0)
                if np.abs(th2np(net.expH[0]) - E_lb) < 1e-4:
                    self.save(net, os.path.join(config.comment, 'net.data'))
                    break
            if epoch>0 and epoch % config.freq_adjust == 0 :
                lr = np.maximum( lr * config.lr_decay, config.lr_end)
                optimizer = lr_scheduler(optimizer, lr)
                self.save(net, os.path.join(config.comment, 'net.data'))
        writer.close()
        return np.array(E_history[-50:]).mean()

    def load(self, directory=None):
        self.net.load_state_dict(th.load(directory, map_location = self.device))
        logging.info('=======================================================')
        logging.info('model has been loaded..')
        logging.info('=======================================================')
    
    def save(self, net, directory=None):
        th.save(net.state_dict(), directory)
        logging.info('=======================================================')
        logging.info('model has been saved..')
        logging.info('=======================================================')

if __name__ == "__main__":
    ham = hamiltonian(atom='H 0 0 0; H 0 0 0.7414', basis_type ="ccpvdz")
    net_paras = dict2obj({
        "amp_hidden":32,
        "phase_hidden":32,
        # "dilation_depth":5,
        'device':th.device("cuda" if th.cuda.is_available() else "cpu"),
        'dtype': th.float32,
    })
    optim_paras = dict2obj({
    "comment": 'H2hd32EXact',
    "epsilon": 0.001,  ## threshold for coupling constant
    "nepoch":5000,
    "batchsize": 500,
    "lr":3e-4,
    "freq_display":1,
    })
    ### training
    vmc_optimizer = VMC(ham, net_paras)
    vmc_optimizer.train(optim_paras, exact=True)
