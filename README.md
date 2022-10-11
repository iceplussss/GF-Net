# Generative Fermionic Neural Network State (GF-Net)

GF-Net is a deep neural network ansatz for many-body electronic states within second quantization framework.  It can be used for calculating electronic ground states of molecular systems with variational Monte Carlo (VMC) algorithm.

Unlike other VMC methods, GF-Net can generate stochastic samples in parallel, as a substitute of sequential Markov Chain Monte Carlo (MCMC) method.  When running on GPUs, GFNet is expected to be much more efficient than MCMC on stochastic sampling. 



## Structure of GF-Net

A GF-Net is essentially a deep autoregressive neural network. It is constructed in an iterative way:
<img width="700" alt="image" src="https://user-images.githubusercontent.com/21097054/194985954-1d1f1929-3d33-4714-b3cc-ae97a14e7c38.png">

where $θ_i$ is a real function satisfying the normalization condition

The GF-Net is similar to the WaveNet (*Oord, Aaron van den, et al. arXiv preprint arXiv:1609.03499 (2016)*) but specially adapted for representing anti-symmetric Fermionic many-body quantum states. The figure below is a schematic representation of GF-Net.

<img width="800" alt="image" src="https://user-images.githubusercontent.com/21097054/194985610-338d3613-efb0-432e-8692-5e25ed28e7f2.png">

## Parallel Sampling
Similar to the neural autoregressive quantum state (*Sharir, Or, et al. Phys. Rev. Lett. 124.2 (2020): 020503*), GF-Net allows the parallel sampling according to the probability distribution represented by itself. The figure below sketches the sampling procedure. 
<img width="800" alt="image" src="https://user-images.githubusercontent.com/21097054/194985904-84880bb2-770c-4b55-9434-0f100a730a9e.png">

## Benchmark

The ground states of H-H and Li-H are used to benchmark the accuracy of GF-Net:
<img width="900" alt="image" src="https://user-images.githubusercontent.com/21097054/194989256-76ed4798-4d49-4bdb-9259-e0fc58335cb1.png">

The results of GF-Net are in agreement with the full configuration interaction (FCI) results, significantly better than the Hartree Fock (HF) results.

## Example
python main.py
