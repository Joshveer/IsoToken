"""
Start the inference service. SPEC: use vLLM (or TGI) as the serving engine.
No custom transformer, scheduler, LoRA math, or serving engine.

This script launches vLLM's OpenAI-compatible server. Run it when you need
a live inference endpoint (e.g. integration tests or local dev). For unit
tests, the test suite uses a stub server; see tests/conftest.py.
"""
import subprocess
import sys

# SPEC: single base model (e.g. meta-llama/Meta-Llama-3-70B)
MODEL = "meta-llama/Meta-Llama-3-70B"
PORT = 8000


def main():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--port",
            str(PORT),
        ],
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
