"""Experiment runner: train, benchmark, and report."""

__all__ = ["run_experiment"]


def __getattr__(name):
    # Lazy import: keeps `python -m dhl.experiments.runner` warning-free,
    # since importing the package no longer imports the runner module first.
    if name == "run_experiment":
        from dhl.experiments.runner import run_experiment

        return run_experiment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
