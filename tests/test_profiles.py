import textwrap

from mbench import paths, profiles


def setup_files(tmp_path, monkeypatch, script_body="exec .venv/bin/sglang serve --model-path /models/big"):
    script = tmp_path / "serve.sh"
    script.write_text(script_body)
    swap_config = tmp_path / "config.yaml"
    swap_config.write_text(textwrap.dedent(f"""
        macros:
          llama: /opt/llama/llama-server --port ${{PORT}}
        models:
          big-sglang:
            name: "Big · SGLang"
            cmd: /usr/bin/env PORT=${{PORT}} {script}
            aliases: [big]
          small-gguf:
            cmd: ${{llama}} -m /models/small.gguf
    """))
    omp = tmp_path / "models.yml"
    omp.write_text(textwrap.dedent("""
        providers:
          llama-swap:
            modelOverrides:
              big-sglang:
                contextWindow: 131072
                compat:
                  thinkingFormat: openai
              small-gguf:
                compat:
                  thinkingFormat: qwen-chat-template
    """))
    user = tmp_path / "profiles.toml"
    user.write_text('[big-sglang]\nhf_id = "org/big"\nquantization = "MXFP4"\n')
    monkeypatch.setattr(paths, "LLAMA_SWAP_CONFIG", swap_config)
    monkeypatch.setattr(paths, "OMP_MODELS", omp)
    monkeypatch.setattr(paths, "PROFILES", user)
    return script


def test_resolve_merges_llama_swap_omp_and_user_profile(tmp_path, monkeypatch):
    setup_files(tmp_path, monkeypatch)
    profile = profiles.resolve("big")
    assert profile.id == "big-sglang"
    assert (profile.engine, profile.thinking, profile.context) == ("sglang", "openai", 131072)
    assert (profile.hf_id, profile.quantization) == ("org/big", "MXFP4")
    small = profiles.resolve("small-gguf")
    assert (small.engine, small.thinking, small.hf_id) == ("llama.cpp", "qwen", None)


def test_editing_the_launcher_changes_the_fingerprint(tmp_path, monkeypatch):
    script = setup_files(tmp_path, monkeypatch)
    before = profiles.resolve("big-sglang").fingerprint
    script.write_text("exec .venv/bin/sglang serve --model-path /models/big --mem-fraction-static 0.9")
    assert profiles.resolve("big-sglang").fingerprint != before


def test_unknown_model_lists_the_known_ones(tmp_path, monkeypatch):
    setup_files(tmp_path, monkeypatch)
    try:
        profiles.resolve("nope")
    except KeyError as error:
        assert "big-sglang" in str(error)
    else:
        raise AssertionError("expected KeyError")
