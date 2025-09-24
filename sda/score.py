r"""Score modules"""

import math
import torch
import torch.nn as nn

from torch import Size, Tensor
from tqdm import tqdm
from typing import *
from zuko.utils import broadcast

from .nn import *
from functools import partial


class TimeEmbedding(nn.Sequential):
    r"""Creates a time embedding.

    Arguments:
        features: The number of embedding features.
    """

    def __init__(self, features: int):
        super().__init__(
            nn.Linear(32, 256),
            nn.SiLU(),
            nn.Linear(256, features),
        )

        self.register_buffer('freqs', torch.pi * torch.arange(1, 16 + 1))

    def forward(self, t: Tensor) -> Tensor:
        t = self.freqs * t.unsqueeze(dim=-1)
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        return super().forward(t)


class ScoreNet(nn.Module):
    r"""Creates a score network.

    Arguments:
        features: The number of features.
        context: The number of context features.
        embedding: The number of time embedding features.
    """

    def __init__(self, features: int, context: int = 0, embedding: int = 16, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = ResMLP(features + context + embedding, features, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        t = self.embedding(t)

        if c is None:
            x, t = broadcast(x, t, ignore=1)
            x = torch.cat((x, t), dim=-1)
        else:
            x, t, c = broadcast(x, t, c, ignore=1)
            x = torch.cat((x, t, c), dim=-1)

        return self.network(x)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        context: The number of context channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels: int, context: int = 0, embedding: int = 64, **kwargs):
        super().__init__()

        self.embedding = TimeEmbedding(embedding)
        self.network = UNet(channels + context, channels, embedding, **kwargs)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        dims = self.network.spatial + 1

        if c is None:
            y = x
        else:
            y = torch.cat(broadcast(x, c, ignore=dims), dim=-dims)

        y = y.reshape(-1, *y.shape[-dims:])
        t = t.reshape(-1)
        t = self.embedding(t)

        return self.network(y, t).reshape(x.shape)


class MCScoreWrapper(nn.Module):
    r"""Disguises a `ScoreUNet` as a score network for a Markov chain."""

    def __init__(self, score: nn.Module):
        super().__init__()

        self.score = score

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # TODO
    ) -> Tensor:
        return self.score(x.transpose(1, 2), t, c).transpose(1, 2)


class MCScoreNet(nn.Module):
    r"""Creates a score network for a Markov chain.

    Arguments:
        features: The number of features.
        context: The number of context features.
        order: The order of the Markov chain.
    """

    def __init__(self, features: int, context: int = 0, order: int = 1, **kwargs):
        super().__init__()

        self.order = order

        if kwargs.get('spatial', 0) > 0:
            build = ScoreUNet
        else:
            build = ScoreNet

        self.kernel = build(features * (2 * order + 1), context, **kwargs)

    def forward(
        self,
        x: Tensor,  # (B, L, C, H, W)
        t: Tensor,  # ()
        c: Tensor = None,  # (C', H, W)
    ) -> Tensor:
        x = self.unfold(x, self.order)
        s = self.kernel(x, t, c)
        s = self.fold(s, self.order)

        return s

    @staticmethod
    @torch.jit.script_if_tracing
    def unfold(x: Tensor, order: int) -> Tensor:
        x = x.unfold(1, 2 * order + 1, 1)
        x = x.movedim(-1, 2)
        x = x.flatten(2, 3)

        return x

    @staticmethod
    @torch.jit.script_if_tracing
    def fold(x: Tensor, order: int) -> Tensor:
        x = x.unflatten(2, (2 * order  + 1, -1))

        return torch.cat((
            x[:, 0, :order],
            x[:, :, order],
            x[:, -1, -order:],
        ), dim=1)


class VPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """

        x = torch.randn(shape + self.shape).to(self.device)
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)

        err = (self.eps(x, t, c) - eps).square()

        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()


class ScaledVPSDE(nn.Module):
    r"""Creates a noise scheduler for the variance preserving (VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = 1 - \alpha(t)^2 + \eta^2

    Arguments:
        eps: A noise estimator :math:`\epsilon_\phi(x, t)`.
        shape: The event shape.
        alpha: The choice of :math:`\alpha(t)`.
        eta: A numerical stability term.
        scale: float = 1.0 # Added to deal with z ↦ x scaling
    """

    def __init__(
        self,
        eps: nn.Module,
        shape: Size,
        alpha: str = 'cos',
        eta: float = 1e-3,
        scale: float = 1.0 # Added to deal with z ↦ x scaling
    ):
        super().__init__()

        self.eps = eps
        self.shape = shape
        self.dims = tuple(range(-len(shape), 0))
        self.eta = eta
        self.scale = scale

        if alpha == 'lin':
            self.alpha = lambda t: 1 - (1 - eta) * t
        elif alpha == 'cos':
            self.alpha = lambda t: torch.cos(math.acos(math.sqrt(eta)) * t) ** 2
        elif alpha == 'exp':
            self.alpha = lambda t: torch.exp(math.log(eta) * t**2)
        else:
            raise ValueError()

        self.register_buffer('device', torch.empty(()))

    def mu(self, t: Tensor) -> Tensor:
        return self.alpha(t)

    def sigma(self, t: Tensor) -> Tensor:
        return self.scale * (1 - self.alpha(t) ** 2 + self.eta ** 2).sqrt()

    def forward(self, x: Tensor, t: Tensor, train: bool = False) -> Tensor:
        r"""Samples from the perturbation kernel :math:`p(x(t) | x)`."""

        t = t.reshape(t.shape + (1,) * len(self.shape))

        eps = torch.randn_like(x)
        x = self.mu(t) * x + self.sigma(t) * eps

        if train:
            return x, eps
        else:
            return x

    def sample(
        self,
        shape: Size = (),
        c: Tensor = None,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
    ) -> Tensor:
        r"""Samples from :math:`p(x(0))`.

        Arguments:
            shape: The batch shape.
            c: The optional context.
            steps: The number of discrete time steps.
            corrections: The number of Langevin corrections per time steps.
            tau: The amplitude of Langevin steps.
        """

        z = torch.randn(shape + self.shape).to(self.device)
        x = z * self.scale
        x = x.reshape(-1, *self.shape)

        time = torch.linspace(1, 0, steps + 1).to(self.device)
        dt = 1 / steps

        with torch.no_grad():
            for t in tqdm(time[:-1], ncols=88):
                # Predictor
                r = self.mu(t - dt) / self.mu(t)
                x = r * x + (self.sigma(t - dt) - r * self.sigma(t)) * self.eps(x, t, c)

                # Corrector
                for _ in range(corrections):
                    z = torch.randn_like(x)
                    eps = self.eps(x, t - dt, c)
                    delta = tau / eps.square().mean(dim=self.dims, keepdim=True)

                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(t - dt)

        return x.reshape(shape + self.shape)

    def loss(self, x: Tensor, c: Tensor = None, w: Tensor = None) -> Tensor:
        r"""Returns the denoising loss."""

        t = torch.rand(x.shape[0], dtype=x.dtype, device=x.device)
        x, eps = self.forward(x, t, train=True)

        err = (self.eps(x, t, c) - eps).square()

        if w is None:
            return err.mean()
        else:
            return (err * w).mean() / w.mean()
            

class SubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-variance preserving (sub-VP) SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t)^2 + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) ** 2 + self.eta


class SubSubVPSDE(VPSDE):
    r"""Creates a noise scheduler for the sub-sub-VP SDE.

    .. math::
        \mu(t) & = \alpha(t) \\
        \sigma(t)^2 & = (1 - \alpha(t) + \eta)^2
    """

    def sigma(self, t: Tensor) -> Tensor:
        return 1 - self.alpha(t) + self.eta


# class DPSGaussianScore(nn.Module):
#     r"""Creates a score module for Gaussian inverse problems.

#     .. math:: p(y | x) = N(y | A(x), Σ)

#     References:
#         | Diffusion Posterior Sampling for General Noisy Inverse Problems (Chung et al., 2022)
#         | https://arxiv.org/abs/2209.14687

#     Note:
#         This module returns :math:`-\sigma(t) s(x(t), t | y)`.
#     """

#     def __init__(
#         self,
#         y: Tensor,
#         A: Callable[[Tensor], Tensor],
#         sde: VPSDE,
#         zeta: float = 1.0,
#     ):
#         super().__init__()

#         self.register_buffer('y', y)

#         self.A = A
#         self.sde = sde
#         self.zeta = zeta

#     def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
#         mu, sigma = self.sde.mu(t), self.sde.sigma(t)

#         with torch.enable_grad():
#             x = x.detach().requires_grad_(True)

#             eps = self.sde.eps(x, t)
#             x_ = (x - sigma * eps) / mu
#             err = (self.y - self.A(x_)).square().sum()

#         s, = torch.autograd.grad(err, x)
#         s = -s * self.zeta / err.sqrt()

#         return eps - sigma * s


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
        guidance_strength: float = 1.0, # Added to control the strength of the score guidance
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.sde = sde
        self.detach = detach
        self.guidance_strength = guidance_strength

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err ** 2 / var).sum() / 2
            dim = tuple(range(1, err.ndim))
            # log_p = - torch.linalg.vector_norm(err, dim=dim)

        s, = torch.autograd.grad(log_p, x)

        return eps - self.guidance_strength*sigma * s


class DPSGaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
        scale: bool = False, # Use scaling in original DPS paper
        guidance_strength: float = 1.0, # Added to control the strength of the score guidance
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))
        self.register_buffer('gamma', torch.as_tensor(gamma))

        self.A = A
        self.sde = sde
        self.detach = detach
        self.scale = scale
        self.guidance_strength = guidance_strength

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_)
            var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

            if self.scale:
                dim = tuple(range(1, err.ndim))
                log_p = -torch.linalg.vector_norm(err, dim=dim)
            else:
                log_p = -(err ** 2 / var).sum() / 2

        s, = torch.autograd.grad(log_p, x)

        return eps - self.guidance_strength * sigma * s


################################# MMPS Guidance #################################
# Linear system solvers (moved from sda/utils.py)

def cg(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system Ax = b with conjugate gradient (CG) iterations."""
    if dtype is None:
        dtype = torch.float64

    epsilon = torch.finfo(dtype).smallest_normal

    if x0 is None:
        x = torch.zeros_like(b)
        r = b
    else:
        x = x0
        r = b - A(x0)

    x = x.to(dtype)
    r = r.to(dtype)
    rr = torch.einsum("...i,...i", r, r)
    p = r

    for _ in range(iterations):
        Ap = A(p.to(b)).to(dtype)
        pAp = torch.einsum("...i,...i", p, Ap)
        alpha = rr / torch.clip(pAp, min=epsilon)
        x_ = x + alpha[..., None] * p
        r_ = r - alpha[..., None] * Ap
        rr_ = torch.einsum("...i,...i", r_, r_)
        beta = rr_ / torch.clip(rr, min=epsilon)
        p_ = r_ + beta[..., None] * p

        x, r, rr, p = x_, r_, rr_, p_

    return x.to(b)


def gmres(
    A: Callable[[Tensor], Tensor],
    b: Tensor,
    x0: Optional[Tensor] = None,
    iterations: int = 1,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    r"""Solves a linear system Ax = b with generalized minimal residual (GMRES) iterations."""
    if dtype is None:
        dtype = torch.float64

    epsilon = torch.finfo(dtype).smallest_normal

    if x0 is None:
        r = b
    else:
        r = b - A(x0)

    r = r.to(dtype)

    def normalize(x):
        norm = torch.linalg.vector_norm(x, dim=-1)
        x = x / torch.clip(norm[..., None], min=epsilon)
        return x, norm

    def rotation(a, b):
        c = torch.clip(torch.sqrt(a * a + b * b), min=epsilon)
        return a / c, -b / c

    V = [None for _ in range(iterations + 1)]
    B = [None for _ in range(iterations + 1)]
    H = [[None for _ in range(iterations)] for _ in range(iterations + 1)]
    cs = [None for _ in range(iterations)]
    ss = [None for _ in range(iterations)]

    V[0], B[0] = normalize(r)

    for j in range(iterations):
        v = V[j].to(b)
        w = A(v).to(dtype)

        # Arnoldi
        for i in range(j + 1):
            H[i][j] = torch.einsum("...i,...i", w, V[i])
            w = w - H[i][j][..., None] * V[i]
        w, w_norm = normalize(w)
        H[j + 1][j] = w_norm
        V[j + 1] = w

        # Givens rotations
        for i in range(j):
            tmp = cs[i] * H[i][j] - ss[i] * H[i + 1][j]
            H[i + 1][j] = cs[i] * H[i + 1][j] + ss[i] * H[i][j]
            H[i][j] = tmp

        cs[j], ss[j] = rotation(H[j][j], H[j + 1][j])
        H[j][j] = cs[j] * H[j][j] - ss[j] * H[j + 1][j]

        # Update residual vector
        B[j + 1] = ss[j] * B[j]
        B[j] = cs[j] * B[j]

        # Fill trailing zeros
        for i in range(j + 1, iterations + 1):
            H[i][j] = torch.zeros_like(H[j][j])

    V, B, H = V[:-1], B[:-1], H[:-1]
    V = torch.stack(V, dim=-2)
    B = torch.stack(B, dim=-1)
    H = torch.stack([torch.stack(Hi, dim=-1) for Hi in H], dim=-2)

    y = torch.linalg.solve_triangular(
        H + epsilon * torch.eye(iterations, dtype=dtype, device=H.device),
        B.unsqueeze(dim=-1),
        upper=True,
    ).squeeze(dim=-1)

    if x0 is None:
        x = torch.einsum("...ij,...i", V, y)
    else:
        x = x0 + torch.einsum("...ij,...i", V, y)

    return x.to(b)


class MMPSGaussianScore(nn.Module):
    def __init__(
        self,
        y: Tensor,
        observation_fn: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde: VPSDE,
        solver: str="gmres",
        iterations: int=1,        
    ):
        super().__init__()

        self.register_buffer('y', y)
        self.register_buffer('std', torch.as_tensor(std))

        self.observation_fn = observation_fn
        self.sde = sde
        if solver == "cg":
            self.solver = partial(cg, iterations=iterations)
        elif solver == "gmres":
            self.solver = partial(gmres, iterations=iterations)

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            eps = self.sde.eps(x, t, c)
            x_ = (x - sigma * eps) / mu
            y_hat = self.observation_fn(x_)

            def A(v):
                if hasattr(torch, "func") and hasattr(torch.func, "jvp"):
                    return torch.func.jvp(self.observation_fn, (x_.detach(),), (v,))[1]
                else:
                    return torch.autograd.functional.jvp(self.observation_fn, (x_.detach(),), (v,), create_graph=False)[1]
            
            def At(v):
                return torch.autograd.grad(y_hat, x_, v, retain_graph=True)[0]

            def cov_x(v):
                # NOTE Changed this to /alpha since that was in original MMPS code
                return sigma**2/mu * torch.autograd.grad(x_, x, v, retain_graph=True)[0] 

            def cov_y(v):
                return self.std**2 * v + A(cov_x(At(v)))
            
            grad = self.y - y_hat
            grad = self.solver(A=cov_y, b=grad)
            grad = torch.autograd.grad(y_hat, x, grad)[0]
            return eps - sigma * grad
        
        #     var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

        #     log_p = -(err ** 2 / var).sum() / 2
        #     dim = tuple(range(1, err.ndim))
        #     # log_p = - torch.linalg.vector_norm(err, dim=dim)

        # s, = torch.autograd.grad(log_p, x)

        # return eps - self.guidance_strength*sigma * s

# class GaussianScoreScaled(nn.Module):
#     r"""Creates a score module for Gaussian inverse problems.

#     .. math:: p(y | x) = N(y | A(x), Σ)

#     Note:
#         This module returns :math:`-\sigma(t) s(x(t), t | y)`.

#     NOTE: Edited to incorporate z ↦ x scaling
#     """

#     def __init__(
#         self,
#         y: Tensor,
#         A: Callable[[Tensor], Tensor],
#         std: Union[float, Tensor],
#         sde: VPSDE,
#         gamma: Union[float, Tensor] = 1e-2,
#         detach: bool = False,
#         scale: float = 1.0,
#     ):
#         super().__init__()

#         self.register_buffer('y', y)
#         self.register_buffer('std', torch.as_tensor(std))
#         self.register_buffer('gamma', torch.as_tensor(gamma))

#         self.A = A
#         self.sde = sde
#         self.detach = detach
#         self.scale = scale

#     def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
#         mu, sigma = self.sde.mu(t), self.sde.sigma(t)
#         sigma *= self.scale

#         if self.detach:
#             eps = self.sde.eps(x / self.scale, t, c)

#         with torch.enable_grad():
#             x = x.detach().requires_grad_(True)

#             if not self.detach:
#                 eps = self.sde.eps(x / self.scale, t, c)

#             x_ = (x - sigma * eps) / mu

#             err = self.y - self.A(x_)
#             var = self.std ** 2 + self.gamma * (sigma / mu) ** 2

#             log_p = -(err ** 2 / var).sum() / 2

#         s, = torch.autograd.grad(log_p, x)

#         return eps - sigma * s

