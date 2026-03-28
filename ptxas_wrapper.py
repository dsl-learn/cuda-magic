#!/usr/bin/env python3
"""ptxas wrapper that intercepts tileiras-invoked ptxas calls and saves PTX inputs.

Usage:
    python intercept_ptxas.py install    # Install the wrapper
    python intercept_ptxas.py uninstall  # Restore the original binary
    python intercept_ptxas.py status     # Show current status

Environment variables:
    PTX_DUMP_DIR  PTX dump directory, default: ./ptx_dumps
"""

import os
import sys
import shutil
from pathlib import Path


def _find_ptxas() -> Path:
    # Prefer the ptxas bundled next to tileiras when available.
    tileiras = shutil.which("tileiras")
    if not tileiras:
        # nvidia-cuda-tileiras pip package installs to nvidia/cu13/bin/
        try:
            import nvidia.cu13 as _cu13
            candidate = Path(_cu13.__path__[0]) / "bin" / "tileiras"
            if candidate.exists():
                tileiras = str(candidate)
        except ImportError:
            pass
    if tileiras:
        candidate = Path(tileiras).resolve().parent / "ptxas"
        if candidate.exists():
            return candidate

    # Fall back to ptxas from PATH.
    ptxas = shutil.which("ptxas")
    if ptxas:
        return Path(ptxas).resolve()

    # sudo often drops /usr/local/cuda/bin from PATH, so check common CUDA roots.
    cuda_roots = []
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            cuda_roots.append(Path(cuda_root))
    cuda_roots.extend(
        [
            Path("/usr/local/cuda"),
            *sorted(Path("/usr/local").glob("cuda-*")),
        ]
    )
    for root in cuda_roots:
        candidate = root / "bin" / "ptxas"
        if candidate.exists():
            return candidate.resolve()

    sys.exit("ERROR: ptxas not found (make sure tileiras or ptxas is in PATH)")


PTXAS = _find_ptxas()
PTXAS_REAL = PTXAS.with_suffix(".real")
DEFAULT_DUMP_DIR = Path(__file__).parent / "ptx_dumps"


def install():
    if PTXAS_REAL.exists():
        print(f"[already installed] real ptxas: {PTXAS_REAL}")
    else:
        shutil.copy2(PTXAS, PTXAS_REAL)
        print(f"[backup] {PTXAS} -> {PTXAS_REAL}")

    # Self-contained wrapper script:
    #   when executed via the shebang, sys.argv[0] is the script path and
    #   sys.argv[1:] are the forwarded ptxas arguments.
    #   Path(__file__).with_suffix('.real') resolves to ptxas.real.
    wrapper = f"""\
#!/usr/bin/env python3
import fcntl, os, re, sys, shutil
from datetime import datetime
from pathlib import Path

REAL = Path(__file__).with_suffix('.real')
DUMP = Path(os.environ.get('PTX_DUMP_DIR', {str(DEFAULT_DUMP_DIR)!r}))
LOG = DUMP / 'ptxas_calls.log'
DUMP.mkdir(parents=True, exist_ok=True)

def _find_arch(args):
    for i, a in enumerate(args):
        if a.startswith('-arch='):
            return a.split('=', 1)[1]
        if a.startswith('--gpu-name='):
            return a.split('=', 1)[1]
        if a in ('-arch', '--gpu-name') and i + 1 < len(args):
            return args[i + 1]
    return 'sm_unknown'


def _normalize_arch(arch):
    if arch.startswith('sm_'):
        return arch
    if arch.startswith('compute_'):
        return 'sm_' + arch.split('_', 1)[1]
    return arch


def _find_entry_name(ptx):
    try:
        text = ptx.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return ptx.stem
    m = re.search(r'\\.visible\\s+\\.entry\\s+([^\\s(]+)', text)
    if not m:
        return ptx.stem
    return re.sub(r'[^0-9A-Za-z_.-]+', '_', m.group(1))


def _reserve_dest(entry, arch):
    with LOG.open('a+', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        lines = f.read().splitlines()
        try:
            seq = int(lines[0].strip()) if lines else 0
            log_lines = lines[1:] if lines else []
        except ValueError:
            seq = 0
            log_lines = []

        dest = DUMP / f'{{seq:03d}}_{{entry}}_{{arch}}.ptx'
        while dest.exists():
            seq += 1
            dest = DUMP / f'{{seq:03d}}_{{entry}}_{{arch}}.ptx'

        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_lines.append(f'{{seq:03d}} {{stamp}} {{dest.name}}')

        f.seek(0)
        f.truncate()
        f.write(f'{{seq + 1}}\\n')
        if log_lines:
            f.write('\\n'.join(log_lines) + '\\n')
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return dest


def _has_kernel_entry(p: Path) -> bool:
    try:
        return bool(re.search(r'\\.visible\\s+\\.entry', p.read_text(encoding='utf-8', errors='ignore')))
    except OSError:
        return False

args = sys.argv[1:]
ptx  = next((Path(a) for a in args if a.endswith('.ptx') and Path(a).exists() and _has_kernel_entry(Path(a))), None)
if ptx:
    arch = _normalize_arch(_find_arch(args))
    entry = _find_entry_name(ptx)
    dest = _reserve_dest(entry, arch)
    shutil.copy2(ptx, dest)
    try:
        with open('/dev/tty', 'w') as _tty:
            print(f'[ptxas-wrapper] saved: {{dest}} with /dev/tty', file=_tty, flush=True)
    except OSError:
        print(f'[ptxas-wrapper] saved: {{dest}}', file=sys.stderr, flush=True)

os.execv(str(REAL), [str(REAL)] + args)
"""
    PTXAS.write_text(wrapper)
    PTXAS.chmod(0o755)
    DEFAULT_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[installed] wrapper -> {PTXAS}")
    print(f"[dump dir]  {DEFAULT_DUMP_DIR}  (override: PTX_DUMP_DIR=/your/path)")


def uninstall():
    if not PTXAS_REAL.exists():
        print("[not installed] nothing to uninstall")
        return
    shutil.copy2(PTXAS_REAL, PTXAS)
    PTXAS_REAL.unlink()
    print(f"[uninstalled] restored: {PTXAS}")


def status():
    ptx_files = list(DEFAULT_DUMP_DIR.glob("*.ptx")) if DEFAULT_DUMP_DIR.exists() else []
    if PTXAS_REAL.exists():
        print("[status] INSTALLED")
        print(f"  wrapper:    {PTXAS}")
        print(f"  real ptxas: {PTXAS_REAL}")
        print(f"  dump dir:   {DEFAULT_DUMP_DIR}")
        print(f"  ptx files:  {len(ptx_files)}")
    else:
        print("[status] NOT INSTALLED")
        print(f"  ptxas:     {PTXAS}")
        print(f"  dump dir:  {DEFAULT_DUMP_DIR}")
        print(f"  ptx files: {len(ptx_files)}")


def _host_find_entry_name(ptx: Path) -> str:
    import re as _re
    try:
        text = ptx.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return ptx.stem
    m = _re.search(r'\.visible\s+\.entry\s+([^\s(]+)', text)
    if not m:
        return ptx.stem
    return _re.sub(r'[^0-9A-Za-z_.-]+', '_', m.group(1))


def _host_reserve_dest(dump_dir: Path, entry: str, arch: str) -> Path:
    import fcntl
    from datetime import datetime
    log = dump_dir / "ptxas_calls.log"
    with log.open('a+', encoding='utf-8') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        lines = f.read().splitlines()
        try:
            seq = int(lines[0].strip()) if lines else 0
            log_lines = lines[1:] if lines else []
        except ValueError:
            seq = 0
            log_lines = []
        dest = dump_dir / f"{seq:03d}_{entry}_{arch}.ptx"
        while dest.exists():
            seq += 1
            dest = dump_dir / f"{seq:03d}_{entry}_{arch}.ptx"
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_lines.append(f"{seq:03d} {stamp} {dest.name}")
        f.seek(0)
        f.truncate()
        f.write(f"{seq + 1}\n")
        if log_lines:
            f.write('\n'.join(log_lines) + '\n')
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return dest


def cutedsl(script_args: list[str]):
    """Run a script with CUTE_DSL_KEEP_PTX enabled, dumping PTX to DEFAULT_DUMP_DIR."""
    if not script_args:
        sys.exit("Usage: ptxas_wrapper.py cutedsl <script.py> [args...]")
    import subprocess, tempfile
    dump_dir = Path(os.environ.get("PTX_DUMP_DIR", DEFAULT_DUMP_DIR))
    dump_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        env = os.environ.copy()
        env["CUTE_DSL_KEEP_PTX"] = "1"
        env["CUTE_DSL_NO_CACHE"] = "1"
        env["CUTE_DSL_DUMP_DIR"] = tmp
        result = subprocess.run([sys.executable] + script_args, env=env)
        for raw in sorted(Path(tmp).glob("*.ptx")):
            arch_part = raw.stem.rsplit('.', 1)[-1] if '.' in raw.stem else 'sm_unknown'
            entry = _host_find_entry_name(raw)
            dest = _host_reserve_dest(dump_dir, entry, arch_part)
            shutil.copy2(raw, dest)
            print(f"[ptxas-wrapper] saved: {dest}")
    sys.exit(result.returncode)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "cutedsl":
        cutedsl(sys.argv[2:])
    else:
        {"install": install, "uninstall": uninstall, "status": status}.get(
            cmd, lambda: sys.exit(f"Usage: {sys.argv[0]} {{install|uninstall|status|cutedsl}}")
        )()
