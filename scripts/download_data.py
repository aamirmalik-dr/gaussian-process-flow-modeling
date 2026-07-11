"""Data note for the flow-modeling demo.

This project is fully reproducible from a synthetic, divergence-free velocity
field, so no download is required and none is performed by default. The script
exists to keep the project layout consistent and to document how a real ocean
current field could be substituted.

To use real data instead, obtain a public surface-current product (for example
OSCAR or HYCOM sea-surface velocities), regrid the u and v components onto a
regular latitude/longitude grid, and load them into a
``flowgp.field.VelocityField``. The rest of the pipeline (GP reconstruction and
particle advection) is agnostic to how the field was produced.

Usage:
    python scripts/download_data.py
"""

from __future__ import annotations


def main() -> int:
    print(
        "No download needed: this demo uses a synthetic divergence-free velocity "
        "field (flowgp.field.synthetic_flow_field). See the module docstring for "
        "how to substitute a real public ocean-current product."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
