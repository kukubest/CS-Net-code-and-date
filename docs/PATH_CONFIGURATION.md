# Path configuration

`ablation_config.json` is the public, sanitized configuration. Replace only the
six dataset paths with local directories. The unified runner is the recommended
entry point.

Some legacy standalone scripts predate the unified runner and retain Windows
absolute paths from the original laboratory workstation. They are included for
methodological traceability. Search for `G:\CT_RE` before using those scripts
and replace the paths locally. Do not treat those legacy paths as downloadable
data locations.
