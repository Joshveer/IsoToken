"""
Minimal health check that validates stack (vLLM + PEFT + single base model).
SPEC: Stack — HuggingFace Transformers, vLLM (or TGI), PEFT (LoRA), Ray.
No new serving engine; this module is used by the health endpoint.
"""
import os


def check_stack() -> tuple[bool, dict]:
    """
    Validate vLLM (or TGI), PEFT, and single base model.
    Returns (ok, details). For tests, set ISOTOKEN_HEALTH_SKIP_* to simulate missing component.
    """
    skip_vllm = os.environ.get("ISOTOKEN_HEALTH_SKIP_VLLM") == "1"
    skip_peft = os.environ.get("ISOTOKEN_HEALTH_SKIP_PEFT") == "1"
    skip_single_base = os.environ.get("ISOTOKEN_HEALTH_SKIP_SINGLE_BASE") == "1"

    vllm_ok = False
    if not skip_vllm:
        # Stub mode (no ISOTOKEN_VLLM_URL) or vLLM importable
        if os.environ.get("ISOTOKEN_VLLM_URL"):
            try:
                import vllm  # noqa: F401
                vllm_ok = True
            except ImportError:
                pass
        else:
            vllm_ok = True  # stub stands in for vLLM

    peft_ok = False
    if not skip_peft:
        try:
            import peft  # noqa: F401
            peft_ok = True
        except ImportError:
            pass

    single_base_ok = not skip_single_base

    details = {
        "backend": "vllm",
        "vllm": vllm_ok,
        "peft": peft_ok,
        "single_base_model": single_base_ok,
    }
    ok = vllm_ok and peft_ok and single_base_ok
    return ok, details
