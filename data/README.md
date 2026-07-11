# Data

This directory is gitignored (this README aside). No datasets are committed.

This project is fully reproducible from a synthetic, divergence-free velocity
field, so no download is required. `scripts/download_data.py` explains how a real
public ocean-current product (for example OSCAR or HYCOM sea-surface velocities)
could be regridded and loaded into a `flowgp.field.VelocityField` in place of the
synthetic field; the GP reconstruction and advection code is agnostic to the
source of the field.
