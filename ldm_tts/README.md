# LDM-TTS Shared Core

`ldm_tts` is the shared implementation layer for Large Discovery Model
test-time search across the three task families in this repository:

- `nanogpt`: code-edit candidates over `train.py`
- `small_molecule`: molecule reservoirs with BO/EHVI tilted selection
- `antibody`: CDRH3 sequence reservoirs with AntBO acquisition scoring

The package intentionally stays dependency-light. Domain modules keep their own
prompt construction, candidate schemas, GP/acquisition math, and environment
evaluators. The shared layer owns only the cross-task mechanics that should not
be copied three times:

- `ldm_tts.loop.run_budgeted_search`: budget, round numbering, empty-reservoir
  accounting, and early-stop reasons.
- `ldm_tts.trajectory.JsonlTrajectoryRecorder`: JSONL round/event logging plus
  companion JSON artifacts such as `config.json`, `history.json`, and
  `summary.json`.
- `ldm_tts.trajectory.AtomicJsonLog`: atomic JSON document updates for decision
  logs.
- `ldm_tts.scoring`: finite numeric score conversion and ranking helpers.

The task-specific code should adapt into this core rather than reimplementing
these mechanics. Keep domain-specific scoring and proposal logic inside the
task package, and pass those functions into the shared loop or recorders.
