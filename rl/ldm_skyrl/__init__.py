"""SkyRL port of the LDM small-molecule acquisition RL loop.

A second training line beside the existing slime one, not a replacement.  Both
lines are judged by the same measurement: Pareto hypervolume at budget 80 on
G12D and G12C, SFT+RL against SFT-only.

The roadmap, the evidence behind it, and the phase register are in
``skyrl_port_plan.ipynb`` and ``phases.json``.  Nothing in this package touches
a GPU at import time.
"""

from ._deps import PhaseGateError, gate, phase_item

__all__ = ["PhaseGateError", "gate", "phase_item"]
