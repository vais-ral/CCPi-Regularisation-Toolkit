import os, sys
from typing import List, Optional, Tuple

cupy_enabled = True
try:
    import cupy as xp

    try:
        xp.cuda.Device(0).compute_capability

    except xp.cuda.runtime.CUDARuntimeError:
        import numpy as xp

        cupy_enabled = False

except ImportError:

    import numpy as xp

    cupy_enabled = False


def load_cuda_module(
    file: str,
    name_expressions: Optional[List[str]] = None,
    options: Tuple[str, ...] = tuple(),
) -> xp.RawModule:
    """Load a CUDA module file, i.e. a .cuh file, from the file system,
    compile it, and return is as a CuPy RawModule for further
    processing.
    """
    dir = os.path.dirname(os.path.abspath(__file__))
    file = os.path.join(dir, file + ".cuh")
    # insert a preprocessor line directive to assist compiler errors (so line numbers show correctly in output)
    escaped = file.replace("\\", "\\\\")
    code = '#line 1 "{}"\n'.format(escaped)
    with open(file, "r") as f:
        code += f.read()

    return xp.RawModule(
        options=("-std=c++11", *options), code=code, name_expressions=name_expressions
    )
