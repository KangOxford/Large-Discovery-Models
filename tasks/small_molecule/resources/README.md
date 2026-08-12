# Small-Molecule Resources

Versioned evaluator metadata and other redistributable inputs live here.
Generated docking files, caches, trajectories, and reports belong in `../runs/`
and are ignored by Git.

## External G12D model

The G12D joblib artifact is deliberately not distributed in Git, and this
project does not currently document a public download URL. Obtain a compatible
artifact from a trusted project maintainer or train and validate one locally.
You may store it at the conventional ignored path:

```text
tasks/small_molecule/resources/models/best_g12d_model.joblib
```

Alternatively, keep it outside the checkout and configure it with:

```bash
export G12D=/trusted/path/best_g12d_model.joblib
```

The tracked `models/best_g12d_model_metadata.json` records the published
artifact's provenance and SHA-256 digest:

```text
a4c15c1124eced2e8dc80a18fdf94752da106168209d804002b0defbf63986ed
```

Place or copy that metadata file beside the model as
`best_g12d_model_metadata.json` to make the scorer and real-run preflight verify
the declared digest before loading. Custom artifacts without matching checksum
metadata remain the caller's responsibility.

Joblib uses pickle-compatible deserialization, which can execute arbitrary
code. Never load the published artifact or a replacement from an untrusted source. Verify
the checksum after every transfer and review the metadata provenance before
use.

The root MIT license covers repository code. It does not establish that every
third-party dataset or derived model can be redistributed under MIT. The
metadata describes the public direct-assay inputs used to train the published
artifact, but does not currently establish their source URLs and license terms.
That unresolved redistribution status is why the binary is external.
