"""Trajectory transforms: one file per transform, ``transforms/<name>.py`` defining ``<name>(traj, **params)``.

A transform sees each finished training trial before the trainer does and decides which of its steps train, and how
strongly. Write it, then start a run with its name; the run checks it on recorded trials before training starts.
See transforms/README.md; the framework is prl/transforms/ (api.py: what a transform may do; core.py: the runner).
"""
