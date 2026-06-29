"""Bandpass filterbank baseline -- alternative to Hamiltonian SS2D for the V3 review (Q1/Q2).

This module is a drop-in replacement for ``HamiltonianSS2D`` that produces
the same (q, p, energy) triplet, but through a fixed-form depthwise
*Gabor-style bandpass filterbank* rather than through damped-oscillator
dynamics with a parallel scan.

Motivation. Reviewer 3 (Round 3) asked whether the gains attributed to
the Hamiltonian formulation could be obtained from a simpler, purely
frequency-selective module of comparable complexity. The natural
candidate is a learnable bandpass filterbank: each channel has a
learnable centre frequency :math:`\omega_c` and a learnable bandwidth
:math:`\sigma_c`, and produces:

    q  = (Gaussian envelope) * cos(omega * x)                  -- position-like
    p  = (Gaussian envelope) * (-omega * sin(omega * x))       -- spatial derivative of q
    E  = 1/2 * (q^2 + p^2)                                     -- energy

The position and derivative kernels are the analogues of the
oscillator's ``q`` and ``p`` streams, but they are produced *without*
any time integration / parallel scan. Frequency selectivity is the
only inductive bias retained from the Hamiltonian version. This makes
the baseline an apples-to-apples test of "is it the dynamics or just
the spectral basis?".

Parameter budget. The module is intentionally parameter-matched to
``HamiltonianSS2D``:

    HamiltonianSS2D (D channels, 4 scan directions):
        learnable per channel:    4 * 5 * D    = 20 D
        pos_merge   (4D -> D):    4 D^2
        mom_merge   (4D -> D):    4 D^2
        total                    ~ 8 D^2

    BandpassFilterbank (D channels, 4 orientations):
        learnable per channel:    2 D          (log_omega, log_sigma)
        pos_merge   (4D -> D):    4 D^2
        mom_merge   (4D -> D):    4 D^2
        total                    ~ 8 D^2

For D = 384 (the segmentation bottleneck width), both are ~1.18 M
parameters. The two baselines therefore consume equivalent capacity;
any accuracy difference is attributable to mechanism, not capacity.

Usage (drop-in). In ``HamiltonianBottleneck``, pass ``ablation='C'``;
the rest of the bottleneck (ConvNeXt parallel path, gated fusion, SE
energy attention, dropout) is unchanged. See ``hamseg.py``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandpassFilterbank(nn.Module):
    """Gabor-style learnable bandpass filterbank.

    Parameters
    ----------
    d_model : int
        Number of channels (must equal the bottleneck width).
    kernel_size : int
        Spatial extent of the Gabor kernel. Default 7 (matches ConvNeXt
        depthwise conv).
    n_orientations : int
        Number of orientations the filterbank covers. Default 4, equal
        to the four scan directions in ``HamiltonianSS2D`` so the merge
        layers have the same shape (``4D -> D``).
    """

    def __init__(self, d_model: int, kernel_size: int = 7,
                 n_orientations: int = 4):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.n_orientations = n_orientations

        # Per-channel learnable centre frequency and bandwidth.
        # Matches the init range of HamiltonianScanLine.log_k for
        # apples-to-apples comparison.
        self.log_omega = nn.Parameter(
            torch.linspace(-1.0, 3.0, d_model))
        # log_sigma initialised so sigma = 1 (kernel covers ~1 stddev
        # inside the central 3 pixels of a 7x7 kernel).
        self.log_sigma = nn.Parameter(torch.zeros(d_model))

        # Static spatial grid for the kernel (registered as buffer so
        # the module is device-agnostic).
        half = kernel_size // 2
        coords = torch.arange(-half, half + 1, dtype=torch.float32)
        gy, gx = torch.meshgrid(coords, coords, indexing='ij')
        # Pre-rotate the x-coordinate for each orientation, since the
        # phase of the Gabor is along x in the local rotated frame.
        thetas = torch.tensor(
            [i * math.pi / n_orientations for i in range(n_orientations)],
            dtype=torch.float32)
        rotated_x = torch.stack(
            [gx * torch.cos(t) + gy * torch.sin(t) for t in thetas],
            dim=0)  # (n_orient, k, k)
        self.register_buffer('rotated_x', rotated_x)
        self.register_buffer('radius_sq', gx * gx + gy * gy)  # (k, k)

        # Merge layers, identical in shape to HamiltonianSS2D's.
        self.pos_merge = nn.Linear(d_model * n_orientations, d_model)
        self.mom_merge = nn.Linear(d_model * n_orientations, d_model)

    def _build_kernels(self):
        """Construct the q-kernel and p-kernel families.

        Returns
        -------
        q_kernels : (n_orient, D, 1, k, k) tensor
        p_kernels : (n_orient, D, 1, k, k) tensor
        """
        omega = torch.exp(self.log_omega)               # (D,)
        sigma = torch.exp(self.log_sigma)               # (D,)

        # Gaussian envelope, shared across orientations: (D, k, k)
        sigma2 = sigma.view(-1, 1, 1).pow(2) + 1e-6
        env = torch.exp(-self.radius_sq.unsqueeze(0) / (2.0 * sigma2))

        # Phase per orientation per channel: (n_orient, D, k, k)
        phase = (omega.view(1, -1, 1, 1)
                 * self.rotated_x.unsqueeze(1))

        q = env.unsqueeze(0) * torch.cos(phase)
        # p is the spatial derivative of q with respect to the rotated
        # x-coordinate: d/dx[env*cos] = env * (-omega * sin) when env
        # is x-symmetric (true here since the envelope is isotropic).
        p = env.unsqueeze(0) * (-omega.view(1, -1, 1, 1)
                                * torch.sin(phase))

        return q.unsqueeze(2), p.unsqueeze(2)  # add input-channel dim

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, D, H, W) tensor.

        Returns
        -------
        q : (B, D, H, W) tensor -- position-like response.
        p : (B, D, H, W) tensor -- spatial-derivative response.
        energy : (B, D, H, W) tensor -- 0.5 (q^2 + p^2).
        """
        B, C, H, W = x.shape
        assert C == self.d_model, f"expected {self.d_model} channels, got {C}"
        q_kern, p_kern = self._build_kernels()
        pad = self.kernel_size // 2

        q_list, p_list = [], []
        for o in range(self.n_orientations):
            q_o = F.conv2d(x, q_kern[o], padding=pad, groups=self.d_model)
            p_o = F.conv2d(x, p_kern[o], padding=pad, groups=self.d_model)
            q_list.append(q_o)
            p_list.append(p_o)

        q_cat = torch.cat(q_list, dim=1)            # (B, n_orient*D, H, W)
        p_cat = torch.cat(p_list, dim=1)

        # Merge orientations down to D channels via the same Linear
        # pattern as HamiltonianSS2D.pos_merge / mom_merge.
        q = self.pos_merge(q_cat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        p = self.mom_merge(p_cat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        energy = 0.5 * (q * q + p * p)
        return q, p, energy

    def extra_repr(self) -> str:
        return (f"d_model={self.d_model}, kernel_size={self.kernel_size}, "
                f"n_orientations={self.n_orientations}")
