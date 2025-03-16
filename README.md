# Path2AI

Path2AI is a set of documents to learn about AI.


# Requirements

1. install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. install requirements:
```bash
uv add --dev ipykernel
```
3. add juypter kernel:
```bash
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=path2AI
```

4. start:
```bash
uv run --with jupyter jupyter lab
```