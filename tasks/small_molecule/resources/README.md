# Small-Molecule Resources

Versioned evaluator metadata and other redistributable inputs live here.
Generated docking files, caches, trajectories, and reports belong in `../runs/`
and are ignored by Git.

## KRAS G12D model

The reference KRAS G12D activity model and its metadata are located in
`models/`:

```text
tasks/small_molecule/resources/models/best_g12d_model.joblib
tasks/small_molecule/resources/models/best_g12d_model_metadata.json
```

Configure real small-molecule runs to use the included model from the
repository root:

```bash
export G12D="$PWD/tasks/small_molecule/resources/models/best_g12d_model.joblib"
```

The metadata sidecar records the model's training provenance, evaluation
results, and SHA-256 digest:

```text
a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed
```

Keep the metadata beside the joblib file. The scorer and real-run preflight
discover it automatically and verify the declared digest before loading the
model. See [`models/README.md`](models/README.md) for the model summary,
validation metrics, configuration options, and manual verification command.

Joblib uses pickle-compatible deserialization, which can execute arbitrary
code. A matching checksum establishes integrity, not trust. Never load the
included artifact or a replacement from an untrusted source.

The root MIT license covers repository code. It does not establish that every
third-party dataset or derived model can be redistributed under MIT. The
metadata describes the public direct-assay inputs used to train the model;
review its provenance and any applicable source terms before redistributing or
using the artifact outside this project.
