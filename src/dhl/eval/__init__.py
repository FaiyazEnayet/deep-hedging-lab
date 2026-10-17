"""Out-of-time evaluation utilities: chronological splits, contiguous
historical test paths, and paired strategy comparison.

The point of this module is honesty about generalization. A policy
trained on block-bootstrapped 2015-2026 paths and tested on more
bootstrapped paths from the same years has never really left the
building. These helpers enforce a harder standard: train on returns
strictly before a cutoff date, then evaluate on contiguous windows of
what actually happened afterwards, with every strategy seeing the
identical paths so performance gaps are policy gaps, not luck.
"""
