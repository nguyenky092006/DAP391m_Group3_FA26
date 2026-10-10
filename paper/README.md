# Acoustic-cue SER manuscript

Minimal self-contained Springer LNCS paper package.

## Files

- `main.tex`: authoritative LaTeX source.
- `references.bib`: cited bibliography entries.
- `llncs.cls` and `splncs04.bst`: Springer LNCS support files.
- `figures/`: figures referenced by `main.tex`.
- `manuscript.pdf`: verified compiled manuscript.

## Build

Compile from this directory with a LaTeX engine that runs BibTeX, for example:

```text
tectonic -X compile main.tex
```

The verified output uses A4 paper, contains 13 pages including References, and embeds all fonts.

This directory intentionally excludes datasets, checkpoints, predictions, intermediate LaTeX files, review prompts, and local experiment artifacts.
