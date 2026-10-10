"""Tests for the convex risk measures."""

import math

import torch

from dhl.risk import cvar, entropic_risk, variance


def test_variance_of_constant_is_zero():
    assert float(variance(torch.full((100,), 2.0))) == 0.0


def test_cvar_above_mean():
    torch.manual_seed(0)
    L = torch.randn(10000)
    assert float(cvar(L, 0.95)) > float(L.mean())


def test_cvar_of_constant_is_constant():
    L = torch.full((500,), 3.0)
    assert abs(float(cvar(L, 0.9)) - 3.0) < 1e-6


def test_cvar_monotone_in_alpha():
    torch.manual_seed(1)
    L = torch.randn(5000)
    assert float(cvar(L, 0.99)) >= float(cvar(L, 0.9))


def test_entropic_limits():
    torch.manual_seed(2)
    L = torch.randn(20000)
    # gamma -> 0 recovers the mean.
    assert abs(float(entropic_risk(L, 1e-4)) - float(L.mean())) < 0.02
    # Larger gamma penalizes the tail more.
    assert float(entropic_risk(L, 2.0)) > float(entropic_risk(L, 0.5))


def test_risk_measures_differentiable():
    L = torch.randn(256, requires_grad=True)
    for fn in (variance, lambda x: cvar(x, 0.9), lambda x: entropic_risk(x, 1.0)):
        loss = fn(L)
        loss.backward()
        assert L.grad is not None and torch.isfinite(L.grad).all()
        L.grad = None
