""" Solving the electronic ground state with GFNet.

Testing cases
ham = hamiltonian(atom='H 0 0 0; H 0 0 0.741', basis_type ="6-31g")
ham = hamiltonian(atom='H 0 0 0; Li 0 0 1.595', basis_type ="6-31g")
ham = hamiltonian(atom='Be 0 0 0;', basis_type ="6-31g")
ham = hamiltonian(atom='H 0 0 0; F 0 0 0.917', basis_type ="6-31g")
"""

import os
import logging
import torch as th
from hamiltonian import hamiltonian
from train import VMC
from util import dict2obj

# specify the electronic system
ham = hamiltonian(atom='H 0 0 0; H 0 0 0.741', basis_type ="6-31g")

# specify the neural network parameters
net_paras = dict2obj({
    "amp_hidden": 32,
    "phase_hidden": 32,
    'device': th.device("cuda" if th.cuda.is_available() else "cpu"),
    'dtype': th.float32,
})

# specify the training parameters
optim_paras = dict2obj({
    "comment": 'H2_6-31g',
    "epsilon": 5e-4,  ## threshold for coupling constant
    "nepoch": 50000,
    "batchsize": 1024,
    "freq_display": 10,
    "freq_adjust": 1000,
    "lr_begin": 1e-3,
    "lr_decay": 0.95,
    "lr_end": 1e-5,
})

model_dir =  optim_paras.comment
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

logging.basicConfig(filename=os.path.join(model_dir, 'train.log'),
            filemode='w',level=logging.DEBUG, format='%(levelname)-6s %(message)s')

# running the variational Monte Carlo algorithm
vmc_optimizer = VMC(ham, net_paras)
vmc_optimizer.train(optim_paras, exact=False)
