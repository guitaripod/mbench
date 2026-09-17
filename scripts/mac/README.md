# Serving MLX models for a remote run

`mbench` measures a model wherever an OpenAI-compatible server answers for it. A Mac serving MLX weights is the only
way to score the quality of a phone row that runs MLX, because the same weights cannot run on the card.

    scp serve.py pool.py mac:~/Dev/mlx-serve/
    ssh mac 'cd ~/Dev/mlx-serve && nohup .venv/bin/python pool.py ~/Dev/mlx-serve/models/<model> 8 8085 16384 > pool.log 2>&1 &'
    systemd-run --user --unit=mac-tunnel --collect -p Restart=always \
      ssh -N -L 127.0.0.1:8085:127.0.0.1:8085 mac

`serve.py` holds one model and answers one request at a time, clearing MLX's cache after each; `pool.py` runs several
of those and hands each request to a free one, reporting `/props` so a run reads the right number of slots. macOS
blocks inbound connections to a Python process, so the port comes over an SSH tunnel rather than over the network.

Then, in `~/.config/mbench/models.toml`:

    ["<id>-mac"]
    engine_kind = "mlx"
    remote = { base_url = "http://127.0.0.1:8085", model = "/Users/…/models/<model>", device = "MacBook Pro", soc = "M5 Pro", os = "macOS 27.0" }

and name it as the phone row's `twin` so the phone's quality columns come from it.
