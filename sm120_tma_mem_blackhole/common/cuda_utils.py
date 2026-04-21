"""CUDA diagnostics: cuobjdump resolution, cubin extraction, SASS/ELF analysis."""

import torch


def warmup_cuda_context() -> None:
    """Force CUDA context creation so NVML baseline is stable across runs."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    del _
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def make_triton_allocator():
    """Return a Triton allocator backed by torch.empty()."""
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    return alloc_fn



import shutil
import sqlite3
import subprocess
from pathlib import Path


def resolve_cuobjdump() -> str:
    """Prefer Triton's bundled cuobjdump, fall back to PATH."""
    try:
        import triton  # noqa: F401
        p = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
        if p.exists():
            return str(p)
    except ImportError:
        pass
    return shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump"


def dump_cubins(db_path: Path, out_dir: Path) -> list[Path]:
    """Export ELF cubins from a sqlite3 cache database."""
    out_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    if not db_path.exists():
        return paths
    conn = sqlite3.connect(db_path)
    try:
        # Some caches include blob_size, others don't.
        cur = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='cache'"
        )
        row = cur.fetchone()
        if row and "blob_size" in row[0]:
            select = "SELECT key, blob, blob_size FROM cache"
        else:
            select = "SELECT key, blob FROM cache"
        cur = conn.execute(select)
        for i, tpl in enumerate(cur.fetchall()):
            blob = tpl[1]
            if not blob[:4] == b"\x7fELF":
                continue
            key = tpl[0]
            p = out_dir / f"blob_{i}_{key[:12]}.cubin"
            p.write_bytes(blob)
            paths.append(p)
    finally:
        conn.close()
    return paths


def analyze_cubin(path: Path, cuobjdump_path: str | None = None) -> dict:
    """Analyze a cubin: count SASS instructions and ELF syscall symbols.

    Returns:
        {"UTMALDG": int, "UBLKCP": int, "CALL_ABS": int,
         "LDG": int, "STG": int, "syscall_refs": list[str]}
    """
    if cuobjdump_path is None:
        cuobjdump_path = resolve_cuobjdump()

    def _run(flag: str) -> str:
        return subprocess.check_output(
            [cuobjdump_path, flag, str(path)],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace")

    try:
        syms = _run("--dump-elf-symbols")
    except subprocess.CalledProcessError:
        syms = ""
    syscall_refs = [l.strip() for l in syms.splitlines() if "cuda_syscall" in l]

    try:
        sass = _run("--dump-sass")
    except subprocess.CalledProcessError:
        sass = ""

    def _count(needle: str) -> int:
        return sum(1 for l in sass.splitlines() if needle in l)

    return {
        "UTMALDG": _count("UTMALDG"),
        "UBLKCP": _count("UBLKCP"),
        "CALL_ABS": _count("CALL.ABS.NOINC"),
        "LDG": _count(" LDG."),
        "STG": _count(" STG."),
        "syscall_refs": syscall_refs,
    }


def count_sass(
    path: Path,
    needles: dict[str, str],
    cuobjdump_path: str | None = None,
) -> dict[str, int]:
    """Count occurrences of specified SASS substrings in a cubin.

    Args:
        needles: {key: substring}, e.g. {"UTMALDG": "UTMALDG", "LDG.E": " LDG.E"}
    """
    if cuobjdump_path is None:
        cuobjdump_path = resolve_cuobjdump()
    out = subprocess.check_output(
        [cuobjdump_path, "--dump-sass", str(path)],
        stderr=subprocess.STDOUT,
    ).decode("utf-8", errors="replace")
    return {k: sum(1 for l in out.splitlines() if v in l) for k, v in needles.items()}


def verify_no_syscall(
    cache_root: Path,
    pattern: str = "*.cubin",
    cuobjdump_path: str | None = None,
) -> None:
    """Assert that no cubin under cache_root references ``__cuda_syscall_*``.

    Prints findings and raises SystemExit on failure.
    """
    cubins = sorted(cache_root.glob(pattern))
    print(f"\n=== cubin scan ({cache_root}) ===")
    if not cubins:
        print("no cubin found; set CUTE_DSL_KEEP_CUBIN=1 and re-run.")
        return

    if cuobjdump_path is None:
        cuobjdump_path = resolve_cuobjdump()

    bad = False
    for cubin in cubins:
        try:
            out = subprocess.check_output(
                [cuobjdump_path, "--dump-elf-symbols", str(cubin)],
                stderr=subprocess.STDOUT,
            ).decode("utf-8", errors="replace")
        except FileNotFoundError:
            print("cuobjdump not in PATH; skipping ELF symbol scan.")
            return
        except subprocess.CalledProcessError as e:
            print(f"{cubin.name}: cuobjdump failed\n{e.output.decode()}")
            bad = True
            continue
        hits = [ln for ln in out.splitlines() if "cuda_syscall" in ln]
        if hits:
            bad = True
            print(f"{cubin.name}: FOUND syscall refs — would leak on 5090:")
            for h in hits:
                print(f"  {h.strip()}")
        else:
            print(f"{cubin.name}: clean (no __cuda_syscall_* symbols)")
    if bad:
        raise SystemExit(
            "at least one cubin still references cluster-scope TMA syscall"
        )
