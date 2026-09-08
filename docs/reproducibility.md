# Reproducibility notes

The primary run is deterministic at the configured seed to the extent supported by the local software/hardware stack. The code records configuration and environment metadata.

The main computational safeguard is checkpointing. Partial forecast results are periodically written. A restart validates the dataset and configuration, retains the contiguous completed prefix, reconstructs causal ensemble error history, and continues from the remaining forecast origins.

For exact replication, use the frozen dataset and do not change the core configuration between interrupted sessions.
