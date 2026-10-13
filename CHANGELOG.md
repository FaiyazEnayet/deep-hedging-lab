# Changelog

## [Unreleased]

### Fixed
- Look-ahead leakage in the decision-time features (`_vol_features` in `src/dhl/experiments/cppi_learned.py`, `_trail_features` in `src/dhl/experiments/cppi_multi.py`). Both builders included the return from price `i` to price `i+1` in the feature vector used to choose the allocation at time `i`. They now use strictly past returns only, and the empty first-step window yields exactly `0.0`. Added regression tests pinning temporal causality (perturbing price `i+1` leaves features at times `<= i` unchanged) and the exact-zero first step.
- Reran all experiments on the corrected features and updated the README tables. Corrected single-asset 63d learned numbers: E[wealth] 102.00 to 101.70, std 4.28 to 3.63, P(breach) 0.004 to 0.003. Multi-asset and cross-decade numbers moved only in the third decimal. The conclusions are unchanged.
