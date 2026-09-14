# mbenchd

The iOS side of an mbench phone run: llama.cpp's own `llama-server`, built for arm64 iOS and run inside an app, plus a small control server that reports what a phone has instead of a driver — thermal state, the memory the process holds, and the battery.

The app serves two ports on the device:

| Port | What answers | What it is for |
|---|---|---|
| 8080 | `llama-server` itself, unproxied | `/v1/chat/completions`, `/props`, `/slots`, `/metrics` — mbench measures straight against it, so no timing passes through code of ours |
| 8081 | mbenchd | `/mb/health`, `/mb/models`, `/mb/load`, `/mb/unload`, and llama-swap's `/running` and `/unload` |

## Building from Linux

No Mac and no Xcode. Metal is the usual obstacle — `xcrun metal` only exists on macOS — but llama.cpp's `GGML_METAL_EMBED_LIBRARY` embeds the kernels as *source* and compiles them on the device at first load, so a cross-compile needs nothing but clang and the iPhoneOS sysroot that xtool's Darwin SDK already carries.

```
./scripts/build-llama.sh     # cross-compiles llama.cpp into Vendor/lib (about four minutes)
xtool dev                    # builds, signs and installs the app
xtool launch XTL-<hash>.com.midgar.mbenchd
```

`build-llama.sh` reads `LLAMA_DIR` (default `~/llama.cpp`) and builds the `llama-server-impl` target: the server, its context, mtmd, common and ggml, all static. `Package.swift` links every archive it finds in `Vendor/lib`, so the app is exactly the llama.cpp the checkout was on — recorded in `Vendor/llama-commit.txt` and reported by `/mb/health`, which is what a run's stack fingerprint uses.

The first load of any model is slow: the Metal library is compiled from source on the phone. `/mb/load` waits for it, and the time it took comes back as `load_seconds`.

## Running a benchmark

The app must be in the foreground with the screen on — iOS suspends a backgrounded app and its listening sockets go with it. Guided Access holds it there. Keep the phone plugged in and charged to full before starting, and record it: charging heat moves the steady-state number as much as the model does.

Models live in `Documents/models/*.gguf`; `mbench phone push <file>` puts them there over USB. Logs are `Library/Logs/mbenchd.log` (the app) and `Documents/logs/llama-server.log` (the server); `mbench phone logs` pulls both.
