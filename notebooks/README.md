# Notebooks

Marimo notebooks for the weekly team meetings. Each one is a plain Python file,
so it diffs like any other source file and is committed here. The exported HTML
is not committed; regenerate it when you need to share it.

## Environment variables

The notebooks read data from directories that live outside this repository.
Point these at your checkouts before running anything. A notebook that needs a
variable you have not set stops with a message naming the variable rather than
failing somewhere further down with a confusing error.

| Variable | Directory |
| --- | --- |
| `REPRO` | `reproduce-soak-nsch`, the R reference runs and the fixture matrix |
| `PAPER` | `cv-same-other-paper`, the SOAK paper's own repository |
| `MONSOON_OLD` | `Monsoon - ASD3 ML Old`, the surviving matrices from the prior analysis |

Set them in your shell profile so they persist:

    export REPRO="$HOME/path/to/reproduce-soak-nsch"
    export PAPER="$HOME/path/to/cv-same-other-paper"
    export MONSOON_OLD="$HOME/path/to/Monsoon - ASD3 ML Old"

No committed file contains an absolute path to anyone's home directory.

## Running

To open a notebook and present from it:

    uv run marimo edit notebooks/<name>.py

To export a static copy for sharing:

    uv run marimo export html notebooks/<name>.py -o <name>.html

Use `uv run`, not `uvx`. `uvx marimo` runs marimo in an isolated environment
that has neither the project's dependencies nor `nsch_ml` itself, so every
notebook here fails on import under it.

The exported HTML opens in a browser with no server running, which makes it the
right artifact to share after a meeting. Matplotlib figures render in the export
rather than appearing as blank cells; this was checked before the first notebook
was written, and is worth rechecking if the marimo version changes.

## Conventions

Every number in a notebook is computed from a file at run time. Nothing is typed
in from a previous run, including numbers that appear in `docs/`. If a notebook
shows a figure that a script in `analyses/` already produces, it calls that code
rather than reimplementing it, so there is one definition of each quantity.

`notebooks/` is excluded from mypy, because marimo cell functions take their
inputs as parameters in a way that type checking reads as unused arguments.
Ruff still runs, with `ARG001` and `E501` ignored here for the same reason.
That exclusion is another argument for keeping real computation in `analyses/`
and `src/nsch_ml`, where the type checker does apply.

`__marimo__/` holds session caches and is gitignored.
