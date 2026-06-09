# CLI Reference

## `modal run` — Execute functions

```bash
modal run app.py                              # run single entrypoint
modal run app.py::my_func                     # run specific function
modal run --detach app.py::main               # survives local disconnect
modal run -m my_package.my_app                # run from module path
modal run --env staging app.py::main          # use specific environment
modal run --name my-experiment app.py::main   # name this run
```

Key flags:
- `--detach` / `-d` — **Always use for jobs >5 minutes.** Without it, closing your terminal kills the remote.
- `--quiet` / `-q` — suppress progress indicators
- `--timestamps` — add timestamps to log lines
- `--env` / `-e` — target a specific Modal environment
- `-m` — interpret arg as Python module path instead of file path

## `modal deploy` — Persistent services

```bash
modal deploy app.py                           # deploy the app
modal deploy --name my-service app.py         # named deployment
modal deploy --tag v1.2 app.py                # version tag
modal deploy --stream-logs app.py             # stream logs after deploy
```

## `modal serve` — Development with hot-reload

```bash
modal serve app.py   # like deploy, but watches for file changes
```

## `modal shell` — Interactive container

```bash
modal shell                                  # bare Debian shell
modal shell app.py::my_func                  # same image/gpu/volumes as function
modal shell --gpu=A100 --memory=8192         # custom spec
modal shell -c 'nvidia-smi' app.py::train    # run a one-off command
modal shell app.py::MyClass.my_method        # works with modal.Cls
```

## `modal volume` — Persistent storage

```bash
modal volume create my-data
modal volume list                             # list all volumes
modal volume ls my-data                       # list root
modal volume ls my-data /checkpoints/exp1     # list subdir
modal volume put my-data ./local/file.jsonl /data/   # upload
modal volume get my-data /checkpoints/final/ ./out/  # download
modal volume rm my-data /old-experiment/      # delete
modal volume cp my-data /src/ /dst/           # copy within volume
modal volume delete my-data                   # delete entire volume
modal volume rename old-name new-name
```

## `modal secret` — Environment variables

```bash
modal secret list
modal secret create my-secret HF_TOKEN=hf_xxx API_KEY=sk_xxx
modal secret delete my-secret
```

Secrets are injected as env vars inside containers. Reference in Python:
```python
@app.function(secrets=[modal.Secret.from_name("my-secret")])
def my_func():
    import os
    token = os.environ["HF_TOKEN"]
```

## `modal app` — Manage running/stopped apps

```bash
modal app list                               # all apps
modal app list --json                        # machine-readable
modal app logs <app-name>                    # fetch logs
modal app logs -f <app-name>                 # stream logs (follow)
modal app stop <app-name>                    # stop and terminate
modal app history <app-name>                 # deployment history
modal app rollback <app-name>                # rollback to previous
modal app dashboard <app-name>               # open in browser
```

## `modal container` — Running containers

```bash
modal container list
modal container logs <container-id>
```

## `modal config` / `modal profile` / `modal token`

```bash
modal config set default-gpu L4              # set default GPU
modal profile switch production              # switch workspace
modal token set --token-id xxx --token-secret yyy
```

## Tips

- `modal run` with `--detach` is the primary workflow for training. Combine with `modal app logs -f <name>` to monitor.
- Volume operations are eventually consistent — writes may take a moment to appear in `ls`.
- `modal shell` is excellent for debugging. It gives you the exact same container environment as your function.
- Use `modal app list` to find the app name for `--detach` runs (it's auto-generated from the function name unless you set `--name`).
