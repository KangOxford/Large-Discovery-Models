"""Phase item A2 (and A6): does the vLLM aarch64 wheel run on driver 565?

Criterion, written before the run: `import vllm` succeeds and the model emits
one token on one GH200.

A6 rides along. SkyRL 0.3.0's only inference backend is vLLM, and vllm 0.23.0
requires flashinfer only under `platform_machine == 'x86_64'`, so on aarch64 it
selects some other attention backend. Which one is a fact worth having in a log
rather than rediscovering during a training run.
"""
import os, platform, sys, traceback

FAIL = []

def step(name, fn):
    print(f"\n=== {name} ===", flush=True)
    try:
        fn()
    except Exception:
        FAIL.append(name)
        traceback.print_exc()

def env():
    print("python  ", sys.version.split()[0])
    print("machine ", platform.machine())
    print("node    ", platform.node())

def torch_side():
    import torch
    print("torch          ", torch.__version__)
    print("torch cuda ver ", torch.version.cuda)
    print("cuda available ", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False -- this is the exact "
                           "symptom the CUDA red line predicts for a mismatched build")
    print("device 0       ", torch.cuda.get_device_name(0))
    print("capability     ", torch.cuda.get_device_capability(0))
    free, total = torch.cuda.mem_get_info(0)
    print(f"memory         {free/2**30:.1f} GiB free / {total/2**30:.1f} GiB")
    x = torch.randn(2048, 2048, device="cuda")
    print("matmul check   ", float((x @ x).sum().abs().item()) > 0)

def vllm_import():
    import vllm
    print("vllm           ", vllm.__version__)
    try:
        import flashinfer  # noqa: F401
        print("flashinfer      present")
    except ImportError:
        print("flashinfer      ABSENT (expected on aarch64: vllm 0.23.0 requires it "
              "only under platform_machine == 'x86_64')")

def generate():
    from vllm import LLM, SamplingParams
    model = os.environ["A2_MODEL"]
    print("model          ", model)
    llm = LLM(model=model, max_model_len=512, gpu_memory_utilization=0.45,
              enforce_eager=True, dtype="bfloat16")
    out = llm.generate(["The capital of France is"],
                       SamplingParams(max_tokens=1, temperature=0.0))
    tok = out[0].outputs[0]
    print("token ids      ", list(tok.token_ids))
    print("token text     ", repr(tok.text))
    if not tok.token_ids:
        raise RuntimeError("the engine returned zero tokens")
    print("A2 CRITERION MET: one token emitted on this GH200")

def backend():
    # vLLM logs its chosen backend at engine init; also expose the env override
    # if one was set, so the log says which of the two decided it.
    print("VLLM_ATTENTION_BACKEND env:", os.environ.get("VLLM_ATTENTION_BACKEND", "<unset, auto-selected>"))

def main() -> int:
    for name, fn in [("environment", env), ("torch", torch_side), ("vllm import", vllm_import),
                     ("attention backend (A6)", backend), ("generate one token", generate)]:
        step(name, fn)
    print("\n=== RESULT ===")
    print("FAILED STEPS:", FAIL if FAIL else "none -- A2 PASSES")
    return 1 if FAIL else 0


# The guard is load-bearing, not decoration. vLLM v1 starts its engine core with
# multiprocessing spawn, and spawn re-imports the main module in the child. With
# the body at module level the child re-runs the whole probe, Python catches it
# at _check_not_importing_main, and the failure surfaces as
# "Engine core initialization failed" -- which reads like a vLLM or driver
# problem and is neither.
if __name__ == "__main__":
    sys.exit(main())
