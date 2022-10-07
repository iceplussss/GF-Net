"""A class specifying the Hilbert space and the Hamiltonian.
"""
import numpy as np
from sympy.utilities.iterables import multiset_permutations
import torch as th
from scipy.sparse.linalg import eigsh
import pyscf
from pyscf import cc
from pyscf import gto, scf, ao2mo, fci
from util import bit2integer, integer2bit, th2np, parity, get_h1e, get_eri

class hamiltonian(th.nn.Module):
    '''
    This class defined the Hilbert space and the Hamiltonian, 
    meanwhile, defined the matrix representation (data coming from PyScf) 
    of the Hamiltonian under Hartree-Fock basis.
    Hamiltonian:
        ccs_0: float, nuclear-nuclear coulomb energy
        ccs_1: [norbs x norbs], one body term <i|h1|j>, first nelec of i stands for occupied orbitals, the rest for virtual
        ccs_2: [norbs x norbs x norbs x norbs], two body term <ij|g|kl>
    We have two types of representation of many-body HF-basis:
    (1)Binary representation of many-body HF-basis:
        Tensor with shape (batches, n_orbitals,2)
        (0,0)->(,);   (0,1)->(,up);   (1,0)->(dn,);  (1,1)->(dn,up)
        Examples: 
            (i)RHF-base of 4 electrons is written as ((1,1),(1,1),(0,0),(0,0),\cdots,(0,0))
            (ii)an excited state of (1) can be written as ((1,1),(1,0),(0,1),(0,0),\cdots,(0,0))
    (2)Occupational representation: 
            Tensor with shape (batches, n_orbitals)
            0->(,);   1->(,up);   2->(dn,);  3->(dn,up)
            Example: RHF-base of 4 electrons is written as (3,3,0,0,\cdots,0)
    Transformation between (1) and (2): 
        bin_basis = integer2bit(ocp_basis)
        ocp_basis = bit2integer(bin_basis)  
    
    The binary representation will be mainly used, also as the input to neural network states.
    '''
    def __init__(self, atom, basis_type="ccpvdz"):
        super(hamiltonian, self).__init__()
        ## model definition in PyScf
        self.mol = gto.Mole(atom=atom, basis=basis_type)
        self.mol.build()
        self.mhf = scf.RHF(self.mol).set(verbose=0).run()
        self.mycc = cc.CCSD(self.mhf).run()
        cisolver = fci.FCI(self.mol, self.mhf.mo_coeff)
        self.E_fci = cisolver.kernel()[0]
        self.E_exact = self.mycc.e_tot + self.mycc.ccsd_t()
        self.E_ref = self.mhf.kernel()
        self.orb = self.mhf.mo_coeff ## HF-orbital in ccpvdz representation
        ## countings
        self.occ = self.mhf.mo_occ   ## Hartree state in occupational representation (nbasis)
        if int(self.occ.sum()) % 2 ==0:
            self.charge = int(self.occ.sum())    ## number of electrons
        else:
            raise ValueError('Number of electrons has to be even for paired Hamiltonian')
        self.norbs = self.orb.shape[0]  ## number of orbitals
        self.nocpd = self.charge // 2   ## number of occupied orbitals for each spin
        self.nempt = self.norbs - self.nocpd  ## number of empty orbitals for each spin
        ## coupling constant in 2-rd quantized Born-Oppenheimer Hamiltonians
        # nuclear-nuclear coulomb energy
        self.ccs_0 = self.mol.energy_nuc()   
        # 1-body coupling constant  (norbs,norbs) 
        self.register_buffer('ccs_1',th.tensor(get_h1e(self.mol, self.orb)))
        self.ccs_1[th.abs(self.ccs_1)<1e-8] *= 0
        # self.ccs_1 = th.tensor(get_h1e(self.mol, self.orb),dtype=th.float)
        # 2-body coupling constant  (norbs,norbs,norbs,norbs)     
        self.register_buffer('ccs_2',th.tensor(get_eri(self.mol, self.orb, compact=False, phy_order=True)))
        self.ccs_2[th.abs(self.ccs_2)<1e-8] *= 0
        # asym_sign = th.tril(2*th.ones(self.norbs, self.norbs))-1
        # # add (-1)^[(if i<j)+(if k<l)] 
        # self.ccs_2 = self.ccs_2*asym_sign[:,:,None,None]*asym_sign[None,None,:,:]
    
    @property
    def full_confs(self):
        ''' get all configurations in the binary representation
        Args:
            None
        Returns:
            full_config(Tensor: th.long): tensor with the shape (number of all configurations, norbs).
        '''
        ref_state = np.concatenate([np.ones(self.nocpd),np.zeros(self.nempt)]).astype(int)
        ## all configuration of one spin direction
        polar_conf = th.tensor(
            np.array(list(multiset_permutations(ref_state))),
            dtype=th.long, device=self.ccs_1.device) # (nbatch, norbs)
        nbatch = polar_conf.shape[0]
        full_conf = th.cat([
            polar_conf[:,None,:,None].expand(nbatch, nbatch,self.norbs,1),
            polar_conf[None,:,:,None].expand(nbatch, nbatch,self.norbs,1),
        ], -1)
        return full_conf.reshape(-1, self.norbs,2)
    
    def hf_base_binary(self):
        '''get the binary representation of the HF base and the corresponding auto-regressive amplitude '''
        hf_bin = th.zeros((self.norbs,2),dtype=th.long)
        hf_amplitude = th.zeros((self.norbs, 4))
        hf_bin[:,0] = th.tensor(self.occ>1).to(dtype=th.long)
        hf_bin[:,1] = th.tensor(self.occ>0).to(dtype=th.long)
        hf_ocp = bit2integer(hf_bin,2)
        for idx, ocp in enumerate(hf_ocp):
            hf_amplitude[idx,ocp]=1.0
        return hf_bin.to(device=self.ccs_1.device), hf_amplitude.to(device=self.ccs_1.device)
        
    #========================== self-loop hopping ==========================
    def get_self_hamp(self, center_confs):  
        '''
        The self-loop hopping amplitude, including contributions from 
            (1)1-body coupling that is a self-loop,
            (2)2-body coupling that has a self-loop or is exchange
        Args:
            center_confs(Tensor, th.long): 
                3D tensor of shape (nbatch, norbs,2), in the Binary representation  
        Returns:
            hamp_self, shape = (nbatch)
        ''' 
        N = self.norbs
        rN = th.arange(N)

        ### contribution from  1-body coupling 
        ccs1_loop = th.diag(self.ccs_1)

        hamp_1loop = (center_confs * ccs1_loop[None,:,None]).sum([-1,-2])  # (nbatch)
        
        ### contribution from  2-body coupling self-loop
        ccs2_loop = th.diag(self.ccs_2.reshape(N*N,N*N)).reshape(N,N)  # (norbs,norbs)
        
        ## mask for Pauli-allowed double self-loop of electrons of different spin
        neutral_2loop_mask = center_confs[:,:,0].unsqueeze(-1) \
            * center_confs[:,:,1].unsqueeze(1) # (nbatch,norbs,norbs)
        hamp_2loop_neutral = (neutral_2loop_mask * ccs2_loop.unsqueeze(0)).sum([-1,-2])# (nbatch)

        ## mask for Pauli-allowed double self-loop of electrons of the same spin
        polar_2loop_mask =  (
            center_confs[:,:,0].unsqueeze(-1) * center_confs[:,:,0].unsqueeze(1)
            + center_confs[:,:,1].unsqueeze(-1) * center_confs[:,:,1].unsqueeze(1)
            ) * (1 - th.eye(N).type_as(center_confs)).unsqueeze(0) # (nbatch,norbs,norbs)
        # 1/2 is multiplied because of [ij|ij] and [ji|ji] degeneracy
        hamp_2loop_polar = 0.5 * (polar_2loop_mask * ccs2_loop.unsqueeze(0)).sum([-1,-2])# (nbatch)
        
        ### contribution from  2-body coupling exchange
        ccs2_exchange = th.diag(self.ccs_2.permute(0,1,3,2).reshape(N*N,N*N)).reshape(N,N)
        exchange_mask = polar_2loop_mask
        hamp_exchange = - 0.5 * (exchange_mask*ccs2_exchange.unsqueeze(0)).sum([-1,-2])# (nbatch)

        hamp_self = hamp_1loop + hamp_2loop_neutral + hamp_2loop_polar + hamp_exchange
 
        return hamp_self
    #==========================  one body hopping ==========================
    def get_hamp1_correction(self, active_conf, silent_conf):
        '''
        Get 2-body coupling contribution to 1-body hopping amplitude.
        The correction includes contribution from 2-body coupling that has a self-loop or a relay
        Args:
            active_conf(Tensor, th.long): shape (nbatch, norbs)
            silent_conf(Tensor, th.long): shape (nbatch, norbs)
        Returns:
            hamp_correction(Tensor, th.float): shape (nbatch, norbs, norbs)
        '''
        rN = th.arange(self.norbs)
        ## the mask of Pauli-allowed hoppings, Element (i-j-k) = 1 if j<->k,  otherwise 0
        hopping_pauli = active_conf.unsqueeze(-1) * (1 - active_conf).unsqueeze(1) # (nbatch,norbs,norbs)
        
        ## ========== contribution from self-loop: i->j, k->k
        ccs2_1loop = self.ccs_2.permute(0,2,1,3)[:,:,rN,rN] # (norbs,norbs,*norbs*)
        ## silent side self-loop contribution, shape = (nbatch, norbs,norbs)
        hamp_1loop_silent = (ccs2_1loop[None,:,:,:] * silent_conf[:,None,None,:]).sum(-1)
        ## active side self-loop contribution, shape = (nbatch, norbs,norbs)
        hamp_1loop_active = (ccs2_1loop[None,:,:,:] * active_conf[:,None,None,:]).sum(-1)
        # (i->j,i->i) is forbidden, so take it out
        hamp_1loop_active -= ccs2_1loop[rN,:,rN].unsqueeze(0)
        
        ## ========== contribution from active side relayed hopping: i->k, k->j
        ccs2_relay = self.ccs_2.permute(0,3,1,2)[:,:,rN,rN] # (norbs,norbs,*norbs*)
        hamp_relay_active = (ccs2_relay[None,:,:,:] * active_conf[:,None,None,:]).sum(-1)
        # (i->i,i->j) is forbidden, so take it out
        hamp_relay_active -= ccs2_relay[rN,:,rN].unsqueeze(0)
        
        ## sum up all contribution and enforce Pauli-exclusion
        hamp_correction = (hamp_1loop_silent + hamp_1loop_active - hamp_relay_active) * hopping_pauli
        return hamp_correction

    def get_half_1bd_neighbors(self, center_confs, epsilon, spin_direction = 0): 
        '''
        Non-trivial one-body hopped(all neighbors are different from the center configuration 
        by one electron of the given spin direction) configuration from center configurations.
        The final hopping amplitude includes 
            (1)contribution from 1-body coupling that is not self-loop
            (1)contribution from 2-body coupling that has a self-loop or a relay
        Args:
            center_confs(Tensor, th.long): center configuration
                3D tensor of shape (nbatch, norbs,2), in the Binary representation  
            epsilon(float): threshold for non-trivial coupling
        Returns:
            conf_nbs(Tensor, th.long): neighbors of center_orb due to 1-body correlation 
                3D tensor of shape (n_nbs, norbs,2)
            hamp_nbs(Tensor, th.long):  hopping amplitude to neighbors
                1D tensor of shape (n_nbs)
            idx_nbs(Tensor, th.float): the index of home configuration of each neighbors
                1D tensor of shape (n_nbs)
        In the following, j<->k means the hopping between the two configurations are allowed by Pauli Exclusion
        '''
        N = self.norbs
        ## activa_conf admits hopping, silent_conf rejects hopping but contributes to coupling constant
        active_conf = center_confs[:,:,spin_direction]
        silent_conf = center_confs[:,:,1-spin_direction]
        ##
        active_sign = active_conf.cumsum(1).roll(shifts=1,dims=1)
        active_sign[:,0] *= 0
        sign_matrix = active_sign.unsqueeze(-1) + active_sign.unsqueeze(1) ## normal order sign
        sign_matrix += (1-th.tril(th.ones_like(sign_matrix[0])) ).unsqueeze(0)  ## extra sign comes froms k>i when i->k
        sign_matrix = 1-2*(sign_matrix%2) ## (nbatch, norbs, norbs)
        # create hopping_mask of which Element(i-j-i) = -1, Element(i-j-j) = 1, otherwise 0; shape = (norbs,norbs,norbs)
        # Adding hopping_mask[i,j] to a configuration allowing i<->j would move the 1 from i to j
        eye = th.eye(N, device = active_conf.device, dtype=active_conf.dtype)
        hopping_mask = - eye.unsqueeze(1).expand(N,N,N) + eye.unsqueeze(0).expand(N,N,N)
        
        # Then add hopping_mask to original configuration to create a view of all hoppings
        neighbors = active_conf[:,None,None,:] + hopping_mask.unsqueeze(0) ## (nbatch,norbs,norbs,norbs)
        ## Get the mask of Pauli-allowed hoppings, Element (i-j-k) = 1 if j<->k,  otherwise 0
        hopping_pauli = active_conf.unsqueeze(-1) * (1 - active_conf).unsqueeze(1) # (nbatch,norbs,norbs)
        ## Pauli-allowed hopping amplitude, Element (i-j-k) = ccs[j,k] if j<->k, otherwise 0
        hopping_amp = self.ccs_1.unsqueeze(0) * hopping_pauli
        hopping_amp += self.get_hamp1_correction(active_conf, silent_conf) ## 2-body coupling correction
        
        ## mask of Pauli-allowed hopping with hopping amplitude above threshold
        hopping_allowed = th.abs(hopping_amp) > epsilon  ## (nbatch, norbs, norbs)
        ## get the indication of which neighbor belongs to which center orbital
        n_nbs = hopping_allowed.sum([-1,-2])
        idx_nbs = th.repeat_interleave(
            th.arange(n_nbs.shape[0]).to(device=n_nbs.device), n_nbs)
        ## get the final neighbors  
        conf_nbs =  neighbors[hopping_allowed]
        if spin_direction == 0:
            conf_nbs = th.cat([
                conf_nbs.unsqueeze(-1),
                silent_conf[idx_nbs].unsqueeze(-1),
                ], -1) #shape (n_neighbors, norbs,2)
        else:
            conf_nbs = th.cat([
                silent_conf[idx_nbs].unsqueeze(-1),
                conf_nbs.unsqueeze(-1),
                ], -1) #shape (n_neighbors, norbs,2)
        ## get the final coupling , 1D tensor of shape (n_neighbors)
        hamp_nbs = hopping_amp[hopping_allowed] * sign_matrix[hopping_allowed]
        return conf_nbs, hamp_nbs, idx_nbs
    #==========================  two body hopping ==========================
    def hopmask2shift(self, hop_mask, shift_sign=[1,1]):
        '''
        Turing hopping mask to shift-vector applied to configuration
        Args:
            hop_mask(Tensor, th.long):
                (nbatch,norbs,norbs)
        Returns:
            hop_idx(Tensor, th.long):
                (nbatch*nhop, 4)
            hop_shift(Tensor, th.long):
                (nbatch, n_hop, norbs)
        '''
        nbatch, N, N_ = hop_mask.shape
        hop_idx = th.nonzero(hop_mask).reshape(nbatch,-1,3) # (nbatch, n_hop, 3)
        n_hop= hop_idx.shape[1]
        hop_idx = th.cat([
            hop_idx[:,:,:1],
            th.arange(n_hop,device=hop_mask.device).repeat(nbatch,1).unsqueeze(-1),
            hop_idx[:,:,1:],
            ],-1)  # (nbatch, n_hop, 4)
        hop_idx = hop_idx.reshape(-1,4)
        hop_shift = th.zeros(nbatch, n_hop,N).to(dtype=th.long,device=hop_mask.device)
        hop_shift[hop_idx[:,0],hop_idx[:,1],hop_idx[:,2]] += shift_sign[0]
        hop_shift[hop_idx[:,0],hop_idx[:,1],hop_idx[:,3]] += shift_sign[1] #(nbatch, n_hop, N)
        return hop_idx, hop_shift
    
    def get_polarized_half_2bd_neighbors(self, center_confs, epsilon, spin_direction = 0): 
        '''
        Non-trivial two-body hopped(all neighbors are different from the center configuration 
        by 2 electrons) configuration from center configurations.
        Args:
            center_confs(Tensor, th.long): center configuration
                3D tensor of shape (nbatch, norbs,2), in the Binary representation  
            epsilon(float): threshold for non-trivial coupling
        Returns:
            conf_nbs(Tensor, th.long): neighbors of center_orb due to same-spin 2-body correlation 
                3D tensor of shape (n_nbs, norbs,2)
            hamp_nbs(Tensor, th.long):  hopping amplitude to neighbors
                1D tensor of shape (n_nbs)
            idx_nbs(Tensor, th.float): the index of home configuration of each neighbors
                1D tensor of shape (n_nbs)
        In the following, j<->k means the hopping between the two configurations are allowed by Pauli Exclusion
        '''
        #### utility
        N = self.norbs
        device = center_confs.device
        nbatch = center_confs.shape[0]
        eye = th.eye(N, device=device, dtype=center_confs.dtype)
        active_conf = center_confs[:,:,spin_direction] # (nbatch,norbs)
        silent_conf = center_confs[:,:,1-spin_direction]

        ## pick any two out of occupied orbital and get the associated shift mask
        ##(ij kl)/(ji lk) degeneracy is eliminated here and the exchange will be dealt with later
        hop_from = active_conf.unsqueeze(-1) * active_conf.unsqueeze(1)
        hop_from = th.triu(hop_from) * (1-eye)  ## get rid of degeneracy, (nbatch,N,N)
        hop_from_idx, hop_downshift = self.hopmask2shift(hop_from,shift_sign=[-1,-1])
        n_hopfrom = hop_downshift.shape[1]
        ## pick any two out of empty orbital and get the associated shift mask
        hop_to = (1-active_conf).unsqueeze(-1) * (1-active_conf).unsqueeze(1) 
        hop_to = th.triu(hop_to) * (1-eye)    ## get rid of degeneracy, (nbatch,N,N)
        hop_to_idx, hop_upshift = self.hopmask2shift(hop_to, shift_sign=[1,1])
        n_hopto = hop_upshift.shape[1]
        ## get the hopping mask for all Pauli-allowed hopping, shape=(nbatch, nhopfrom,nhopto,N)
        hop_shift = (hop_downshift.unsqueeze(2) + hop_upshift.unsqueeze(1))
        neighbors = active_conf[:,None,None,:] + hop_shift ## (nbatch,nhopfrom,nhopto,N)
        
        ## get the hopping index (ijkl) for all Pauli-allowed hopping
        hop_from_idx = hop_from_idx.reshape(nbatch,-1,4).unsqueeze(2).expand(nbatch,n_hopfrom,n_hopto,4)
        hop_to_idx = hop_to_idx.reshape(nbatch,-1,4).unsqueeze(1).expand(nbatch,n_hopfrom,n_hopto,4)
        hop_idx = th.cat([hop_from_idx[:,:,:,2:],hop_to_idx[:,:,:,2:]],-1).reshape(-1,4)
        ## get all Pauli-allowed hopping amplitude
        hopping_amp = self.ccs_2[hop_idx[:,0],hop_idx[:,1],hop_idx[:,2],hop_idx[:,3]]
        ## there is a minus sign for exchange!
        hopping_amp -= self.ccs_2.transpose(-1,-2)[hop_idx[:,0],hop_idx[:,1],hop_idx[:,2],hop_idx[:,3]]
        hopping_amp = hopping_amp.reshape(nbatch, n_hopfrom,n_hopto)

        ## mask of Pauli-allowed hopping with hopping amplitude above threshold
        hopping_allowed = th.abs(hopping_amp) > epsilon 
        n_nbs = hopping_allowed.sum([-1,-2])
        idx_nbs = th.repeat_interleave(th.arange(nbatch,device=device), n_nbs)
        ## get the final neighbors  
        conf_nbs =  neighbors[hopping_allowed]  ## (n_nbs, N)
        ## deal with sign at the end
        active_sign = active_conf.cumsum(1).roll(shifts=1,dims=1)
        active_sign[:,0] *= 0 
        active_sign = active_sign[idx_nbs] #(n_nbs, N)
        diff_mask = th.abs(conf_nbs - active_conf[idx_nbs])>0  #(n_nbs, N)
        sign_normal = (active_sign * diff_mask).sum(-1)  ## normal order sign
        sign_normal = (-1)**(sign_normal%2)
        hop_allowed_idx = hop_idx.reshape(nbatch, n_hopfrom, n_hopto,4)[hopping_allowed]
        sign_permute = parity(hop_allowed_idx[:,[0,1,3,2]])## permutation sign
        sign = sign_normal*sign_permute
        ##
        if spin_direction == 0:
            conf_nbs = th.cat([
                conf_nbs.unsqueeze(-1),
                silent_conf[idx_nbs].unsqueeze(-1),
                ], -1) #shape (n_neighbors, norbs,2)
        else:
            conf_nbs = th.cat([
                silent_conf[idx_nbs].unsqueeze(-1),
                conf_nbs.unsqueeze(-1),
                ], -1) #shape (n_neighbors, norbs,2)
        ## get the final coupling , 1D tensor of shape (n_neighbors)
        hamp_nbs = hopping_amp[hopping_allowed] * sign
        return conf_nbs, hamp_nbs, idx_nbs
    
    def get_neutral_2bd_neighbors(self, center_confs, epsilon): 
        '''
        Non-trivial two-body hopped(all neighbors are different from the center configuration 
        by 2 electrons) configuration from center configurations.
        Args:
            center_confs(Tensor, th.long): center configuration
                3D tensor of shape (nbatch, norbs,2), in the Binary representation  
            epsilon(float): threshold for non-trivial coupling
        Returns:
            conf_nbs(Tensor, th.long): neighbors of center_orb due to different-spin 2-body hopping 
                3D tensor of shape (n_nbs, norbs,2)
            hamp_nbs(Tensor, th.long):  hopping amplitude to neighbors
                1D tensor of shape (n_nbs)
            idx_nbs(Tensor, th.float): the index of home configuration of each neighbors
                1D tensor of shape (n_nbs)
        In the following, j<->k means the hopping between the two configurations are allowed by Pauli Exclusion
        '''  
        #### utility
        N = self.norbs
        device = center_confs.device
        nbatch = center_confs.shape[0]
        eye = th.eye(N, device=device, dtype=center_confs.dtype)
        ## calculate sign in the same way as 1bd hopping
        sign = center_confs.cumsum(1).roll(shifts=1,dims=1)
        sign[:,0,:] *= 0
        sign_0 = sign[:,:,0].unsqueeze(-1) + sign[:,:,0].unsqueeze(1)
        sign_0 += (1-th.tril(th.ones_like(sign_0[0])) ).unsqueeze(0)
        sign_1 = sign[:,:,1].unsqueeze(-1) + sign[:,:,1].unsqueeze(1)
        sign_1 += (1-th.tril(th.ones_like(sign_1[0])) ).unsqueeze(0)
        sign_0 = (-1)**(sign_0%2) ## (nbatch, norbs, norbs)
        sign_1 = (-1)**(sign_1%2) ## (nbatch, norbs, norbs)
        ## pick any two out of occupied orbital and get the associated shift mask=(nbatch, n_hop0, norbs)
        hop_0 = center_confs[:,:,0].unsqueeze(-1) * (1-center_confs[:,:,0]).unsqueeze(1)
        sign_0 = sign_0[hop_0>0].reshape(nbatch,-1)
        hop_0_idx, hop_0_shift = self.hopmask2shift(hop_0, shift_sign=[-1,1])
        n_hop0 = hop_0_shift.shape[1]
        ## pick any two out of empty orbital and get the associated shift mask=(nbatch, n_hop1, norbs)
        hop_1 = center_confs[:,:,1].unsqueeze(-1) * (1-center_confs[:,:,1]).unsqueeze(1)
        sign_1 = sign_1[hop_1>0].reshape(nbatch,-1)
        hop_1_idx, hop_1_shift = self.hopmask2shift(hop_1, shift_sign=[-1,1])
        n_hop1 = hop_1_shift.shape[1]
        ## get the hopping mask for all Pauli-allowed hopping, shape=(nbatch, nhopfrom,nhopto,norbs)
        hop_shift = th.cat([
            hop_0_shift[:,:,None,:,None].repeat(1,1,n_hop1,1,1),
            hop_1_shift[:,None,:,:,None].repeat(1,n_hop0,1,1,1),
            ],-1) # (nbatch, n_hop0, n_hop1, N, 2)
        neighbors = center_confs[:,None,None,:,:] + hop_shift 
        
        ## get the hopping index (ijkl) for all Pauli-allowed hopping
        hop_0_idx = hop_0_idx.reshape(nbatch,n_hop0,4).unsqueeze(2).expand(nbatch,n_hop0,n_hop1,4)
        hop_1_idx = hop_1_idx.reshape(nbatch,n_hop1,4).unsqueeze(1).expand(nbatch,n_hop0,n_hop1,4)
        hop_idx = th.cat([hop_0_idx[:,:,:,2:],hop_1_idx[:,:,:,2:]],-1).reshape(-1,4)
        ## get all Pauli-allowed hopping amplitude
        hopping_amp = self.ccs_2.permute(0,2,1,3)[hop_idx[:,0],hop_idx[:,1],hop_idx[:,2],hop_idx[:,3]]
        hopping_amp = hopping_amp.reshape(nbatch, n_hop0, n_hop1) * sign_0.unsqueeze(-1) * sign_1.unsqueeze(1)

        ## mask of Pauli-allowed hopping with hopping amplitude above threshold
        hopping_allowed = th.abs(hopping_amp) > epsilon 
        n_nbs = hopping_allowed.sum([-1,-2])
        idx_nbs = th.repeat_interleave(th.arange(nbatch,device=device), n_nbs)
        ## get the final neighbors  
        conf_nbs =  neighbors[hopping_allowed]  ## (n_nbs, N)
        ## get the final coupling , 1D tensor of shape (n_neighbors)
        hamp_nbs = hopping_amp[hopping_allowed]
        return conf_nbs, hamp_nbs, idx_nbs

    def get_all_neighbors(self, center_confs, epsilon=0.01):
        '''
        get all neighbors of |center_orb> including itself, also the associated hopping amplitude
        Args:
            center_orb(Tensor, th.long): center configuration
                3D tensor of shape (nbatch, norbs,2), in the Binary representation  
            epsilon(float): threshold for non-trivial coupling
        Returns:
            conf_nbs(Tensor, th.long): neighbors of center_orb
                3D tensor of shape (n_nbs, norbs,2)
            hamp_nbs(Tensor, th.long): 
                1D tensor of shape (n_nbs)
            idx_nbs(Tensor, th.float): the index of home configuration of each neighbors
                1D tensor of shape (n_nbs)
        '''
        hamp_self = self.get_self_hamp(center_confs)
        conf_self = center_confs
        idx_self = th.arange(center_confs.shape[0]).to(device=center_confs.device)
        conf_1bdnbs_0, hamp_1bdnbs_0, idx_1bdnbs_0 = self.get_half_1bd_neighbors(center_confs, epsilon, spin_direction = 0)
        conf_1bdnbs_1, hamp_1bdnbs_1, idx_1bdnbs_1 = self.get_half_1bd_neighbors(center_confs, epsilon, spin_direction = 1)
        conf_2bdnbs_0, hamp_2bdnbs_0, idx_2bdnbs_0 = self.get_polarized_half_2bd_neighbors(center_confs, epsilon, spin_direction = 0)
        conf_2bdnbs_1, hamp_2bdnbs_1, idx_2bdnbs_1 = self.get_polarized_half_2bd_neighbors(center_confs, epsilon, spin_direction = 1)
        conf_2bdnbs_n, hamp_2bdnbs_n, idx_2bdnbs_n = self.get_neutral_2bd_neighbors(center_confs, epsilon)
        hamp = [hamp_self,hamp_1bdnbs_0,hamp_1bdnbs_1,hamp_2bdnbs_0,hamp_2bdnbs_1,hamp_2bdnbs_n]
        conf_nbs = th.cat([conf_self,conf_1bdnbs_0,conf_1bdnbs_1,conf_2bdnbs_0,conf_2bdnbs_1,conf_2bdnbs_n],0)
        hamp_nbs = th.cat([hamp_self,hamp_1bdnbs_0,hamp_1bdnbs_1,hamp_2bdnbs_0,hamp_2bdnbs_1,hamp_2bdnbs_n],0)
        idx_nbs = th.cat([idx_self,idx_1bdnbs_0,idx_1bdnbs_1,idx_2bdnbs_0,idx_2bdnbs_1,idx_2bdnbs_n],0)

        return conf_nbs, hamp_nbs, idx_nbs
 

    def exact_diag(self, epsilon=0.001):
        full_confs = self.full_confs
        nbatch = full_confs.shape[0]
        label_confs = bit2integer(full_confs.reshape(nbatch, -1),2*self.norbs)
        M = np.zeros((nbatch,nbatch))
        for i, cf in enumerate(full_confs):
            conf_nbs, hamp_nbs, idx_nbs = self.get_all_neighbors(cf.unsqueeze(0),epsilon)
            label_nbs = bit2integer(conf_nbs.reshape(conf_nbs.shape[0],-1),2*self.norbs)
            for j, hamp in enumerate(hamp_nbs):
                idx_j = th.nonzero(label_confs == label_nbs[j]).sum()
                M[i,idx_j] += hamp
        # e, v = th.symeig(M)
        e,v = eigsh(M, k=1,which='SA')
        Emin = e.min() + self.ccs_0
        # print('==ED==, shape=',nbatch)
        # print('ccsdt exact ground energy:',self.E_exact)
        # print('exact ground energy:',Emin)
        # print('fci ground energy:',self.E_fci)
        return Emin
    def get_relative_Eorb(self):
        Eorb = th.diag(self.ccs_1)*1.0
        Eorb -= Eorb.min()
        Eorb /= Eorb.max()
        return Eorb
    def get_Eoffset(self):
        Eorb = self.get_relative_Eorb()
        E_surface = Eorb[:self.nocpd].max()
        if E_surface == 0 or Eorb.max()==E_surface:
            return Eorb
        else:
            Eoffset = (Eorb - E_surface) / (Eorb.max()-E_surface)
            Eoffset[:self.nocpd] = (Eorb[:self.nocpd] / E_surface) - 1
            ##  (-1,-0.9,...,0(surface), 0.1,...,1)
            scale = 2  ## largest scale exp(scale)
            return Eoffset * scale 


if __name__ == "__main__":
    # ham = hamiltonian(atom='H 0 0 0; H 0 0 0.741', basis_type ="ccpvdz")
    ham = hamiltonian(atom='H 0 0 0;F 0 0 0.917;', basis_type ="6-31g")
    #verify hartree-fock solution
    mhf = scf.RHF(ham.mol).set(verbose=0)
    e_ref = mhf.kernel()
    hf_conf, amp = ham.hf_base_binary()
    e_hf = ham.get_self_hamp(hf_conf.unsqueeze(0))[0] + ham.ccs_0
    if th.abs((e_ref - e_hf)/e_ref)<1e-8:
        print('hf_energy get from this class={} verified'.format(e_ref))
    else:
        print('e_hf',e_hf)
        print('e_ref',e_ref)
        print('e_ref != e_hf')
    print('fci ground energy:',ham.E_fci)
