
import os, sys, time
from pathlib import Path
import torch

CACHE_ROOT = Path(__file__).parent / "cute_cache"
CACHE_ROOT.mkdir(exist_ok=True)
os.environ.setdefault("CUTE_DSL_CACHE_DIR", str(CACHE_ROOT / "mlir_cache"))
os.environ.setdefault("CUTE_DSL_DUMP_DIR", str(CACHE_ROOT))
os.environ.setdefault("CUTE_DSL_KEEP_CUBIN", "1")
if "CUTE_DSL_ARCH" not in os.environ:
    mj, mn = torch.cuda.get_device_capability(0)
    os.environ["CUTE_DSL_ARCH"] = f"sm_{mj}{mn}"

torch.cuda.init()
_ = torch.empty(1, device="cuda"); del _
torch.cuda.synchronize()
print(f"CHILD_PID={os.getpid()}", flush=True)
print("CTX_READY", flush=True)

from cute_tma_copy import TmaIdentityCopy
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

M, N = 1024, 2048
A = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
B = torch.empty_like(A)
a_ct = from_dlpack(A).mark_layout_dynamic(leading_dim=1)
b_ct = from_dlpack(B).mark_layout_dynamic(leading_dim=1)
compiled = cute.compile(TmaIdentityCopy(), a_ct, b_ct)
print("COMPILED", flush=True)

# Fire-and-forget: we know the kernel livelocks on SM120. We only need the
# allocation to trigger, which happens at launch-queue time.
compiled(a_ct, b_ct)
print("LAUNCHED", flush=True)

# Park. Parent will SIGKILL us when it's done sampling.
while True:
    time.sleep(3600)
