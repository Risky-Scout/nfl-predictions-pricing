# Feature manifests

The Stage 2 table is a feature warehouse only. No estimator may consume all
warehouse columns.

Each YAML file defines:

- metadata retained for auditability;
- targets retained outside the feature matrix;
- football-only features;
- market features;
- hard feature-count caps;
- closing/reference horizon classification.

Resolved feature lists and SHA-256 fingerprints are written beside each
compact matrix.
