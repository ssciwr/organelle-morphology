# Contributing

## Development Installation

The recommended dev setup is to clone the repository, then run `uv sync` inside the repo.

## Modify the UI

Run marimo in edit mode to change the UI.
Switch to the app-view to modify the layout.

```
marimo edit --watch src/app/ui.py
```

## Coding Style

* Please use ruff for linting.
* Doc strings should use google style.
* Pre-commit hooks are used to enforce linting.
