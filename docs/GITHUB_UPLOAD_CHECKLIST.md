# GitHub upload checklist

Before publication:

- [ ] Choose and add a software license.
- [ ] Fill in authors, paper title, DOI/preprint URL, and citation metadata.
- [ ] Review all code and result tables for third-party redistribution rights.
- [ ] Confirm the repository contains no patient/private/proprietary data.
- [ ] Confirm `manifests/no_volume_audit.json` reports zero forbidden files.
- [ ] Confirm no file exceeds GitHub's 100 MB hard limit.
- [ ] Keep checkpoints and all 3D volumes in an external archive, not Git.
- [ ] Create a tagged release and attach the ZIP if desired.
- [ ] Add representative figures to the GitHub repository description/README.

Recommended upload scope:

1. Source code and sanitized configuration.
2. Numeric result tables and selected publication figures.
3. Reproducibility/data-availability documentation.
4. File/checksum manifests.

Do not upload the original project directory wholesale.
