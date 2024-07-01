# This work is part of the Core Imaging Library developed by
# Visual Analytics and Imaging System Group of the Science Technology
# Facilities Council, STFC

# Copyright 2019 Daniil Kazantsev
# Copyright 2019 Srikanth Nagella, Edoardo Pasca

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Regularisation functions that are exposed through CuPy API 
    Ver.0.1, Nov. 2023
"""

try:
    import cupy as cp
except ImportError:
    print("CuPy is required, please install")
import numpy as np

from typing import Optional, Tuple
from ccpi.cuda_kernels import load_cuda_module

__all__ = [
    "ROF_TV",
    "PD_TV",
]


def ROF_TV(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 3000,
    time_marching_parameter: Optional[float] = 0.001,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """Total Variation using Rudin-Osher-Fatemi (ROF) explicit iteration scheme to perform edge-preserving image denoising.
       This is a gradient-based algorithm for a smoothed TV term which requires a small time marching parameter and a significant number of iterations.
       Ref: Rudin, Osher, Fatemi, "Nonlinear Total Variation based noise removal algorithms", 1992.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (Optional[int], optional): The number of iterations. Defaults to 3000.
        time_marching_parameter (Optional[float], optional): Time marching parameter, needs to be small to ensure convergance. Defaults to 0.001.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

    Returns:
        cp.ndarray: ROF-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")
    cp.get_default_memory_pool().free_all_blocks()

    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    # initialise CuPy arrays here
    out = data.copy()
    d_D1 = cp.empty(data.shape, dtype=cp.float32, order="C")
    d_D2 = cp.empty(data.shape, dtype=cp.float32, order="C")

    # loading and compiling CUDA kernels:
    module = load_cuda_module("TV_ROF_GPU_kernels")
    if data.ndim == 3:
        data3d = True
        d_D3 = cp.empty(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        D1_func = module.get_function("D1_func3D")
        D2_func = module.get_function("D2_func3D")
        D3_func = module.get_function("D3_func3D")
        TV_kernel = module.get_function("TV_kernel3D")
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        D1_func = module.get_function("D1_func2D")
        D2_func = module.get_function("D2_func2D")
        TV_kernel = module.get_function("TV_kernel2D")

    # perform algorithm iterations
    for iter in range(iterations):
        # calculate differences
        if data3d:
            params1 = (out, d_D1, dx, dy, dz)
        else:
            params1 = (out, d_D1, dx, dy)
        D1_func(grid_dims, block_dims, params1)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params2 = (out, d_D2, dx, dy, dz)
        else:
            params2 = (out, d_D2, dx, dy)
        D2_func(grid_dims, block_dims, params2)
        cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params21 = (out, d_D3, dx, dy, dz)
            D3_func(grid_dims, block_dims, params21)
            cp.cuda.runtime.deviceSynchronize()
        # calculating the divergence and the gradient term
        if data3d:
            params3 = (
                d_D1,
                d_D2,
                d_D3,
                out,
                data,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
                dz,
            )
        else:
            params3 = (
                d_D1,
                d_D2,
                out,
                data,
                cp.float32(regularisation_parameter),
                cp.float32(time_marching_parameter),
                dx,
                dy,
            )
        TV_kernel(grid_dims, block_dims, params3)
        cp.cuda.runtime.deviceSynchronize()
    return out


def PD_TV(
    data: cp.ndarray,
    regularisation_parameter: Optional[float] = 1e-05,
    iterations: Optional[int] = 1000,
    methodTV: Optional[int] = 0,
    nonneg: Optional[int] = 0,
    lipschitz_const: Optional[float] = 8.0,
    gpu_id: Optional[int] = 0,
) -> cp.ndarray:
    """Primal Dual algorithm for non-smooth convex Total Variation functional.
       Ref: Chambolle, Pock, "A First-Order Primal-Dual Algorithm for Convex Problems
       with Applications to Imaging", 2010.

    Args:
        data (cp.ndarray): A 2d or 3d CuPy array.
        regularisation_parameter (Optional[float], optional): Regularisation parameter to control the level of smoothing. Defaults to 1e-05.
        iterations (Optional[int], optional): The number of iterations. Defaults to 1000.
        methodTV (Optional[int], optional): Choose between isotropic (0) or anisotropic (1) case for TV norm.
        nonneg (Optional[int], optional): Enable non-negativity in updates by selecting 1. Defaults to 0.
        lipschitz_const (Optional[float], optional): Lipschitz constant to control convergence.
        gpu_id (Optional[int], optional): A GPU device index to perform operation on. Defaults to 0.

    Returns:
        cp.ndarray: PD-TV filtered CuPy array.
    """
    if gpu_id >= 0:
        cp.cuda.Device(gpu_id).use()
    else:
        raise ValueError("The gpu_device must be a positive integer or zero")

    # with cp.cuda.Device(gpu_id):
    cp.get_default_memory_pool().free_all_blocks()

    input_type = data.dtype

    if input_type != "float32":
        raise ValueError("The input data should be float32 data type")

    # prepare some parameters:
    tau = cp.float32(regularisation_parameter * 0.1)
    sigma = cp.float32(1.0 / (lipschitz_const * tau))
    theta = cp.float32(1.0)
    lt = cp.float32(tau / regularisation_parameter)

    # initialise CuPy arrays here:
    out = data.copy()
    P1 = cp.empty(data.shape, dtype=cp.float32, order="C")
    P2 = cp.empty(data.shape, dtype=cp.float32, order="C")
    d_old = cp.empty(data.shape, dtype=cp.float32, order="C")

    # loading and compiling CUDA kernels:
    module = load_cuda_module("TV_PD_GPU_kernels")
    if data.ndim == 3:
        data3d = True
        P3 = cp.empty(data.shape, dtype=cp.float32, order="C")
        dz, dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_z = dz
        grid_dims = (grid_x, grid_y, grid_z)
        dualPD_kernel = module.get_function("dualPD3D_kernel")
        Proj_funcPD_iso_kernel = module.get_function("Proj_funcPD3D_iso_kernel")
        Proj_funcPD_aniso_kernel = module.get_function("Proj_funcPD3D_aniso_kernel")
        DivProj_kernel = module.get_function("DivProj3D_kernel")
        PDnonneg_kernel = module.get_function("PDnonneg3D_kernel")
        getU_kernel = module.get_function("getU3D_kernel")
    else:
        data3d = False
        dy, dx = data.shape
        # setting grid/block parameters
        block_x = 128
        block_dims = (block_x, 1)
        grid_x = (dx + block_x - 1) // block_x
        grid_y = dy
        grid_dims = (grid_x, grid_y)
        dualPD_kernel = module.get_function("dualPD_kernel")
        Proj_funcPD_iso_kernel = module.get_function("Proj_funcPD2D_iso_kernel")
        Proj_funcPD_aniso_kernel = module.get_function("Proj_funcPD2D_aniso_kernel")
        DivProj_kernel = module.get_function("DivProj2D_kernel")
        PDnonneg_kernel = module.get_function("PDnonneg2D_kernel")
        getU_kernel = module.get_function("getU2D_kernel")

    # perform algorithm iterations
    for iter in range(iterations):
        # calculate differences
        if data3d:
            params1 = (out, P1, P2, P3, sigma, dx, dy, dz)
        else:
            params1 = (out, P1, P2, sigma, dx, dy)
        dualPD_kernel(
            grid_dims, block_dims, params1
        )  # computing the the dual P variable
        cp.cuda.runtime.deviceSynchronize()
        if nonneg != 0:
            if data3d:
                params2 = (out, dx, dy, dz)
            else:
                params2 = (out, dx, dy)
            PDnonneg_kernel(grid_dims, block_dims, params2)
            cp.cuda.runtime.deviceSynchronize()
        if data3d:
            params3 = (P1, P2, P3, dx, dy, dz)
        else:
            params3 = (P1, P2, dx, dy)
        if methodTV == 0:
            Proj_funcPD_iso_kernel(grid_dims, block_dims, params3)
        else:
            Proj_funcPD_aniso_kernel(grid_dims, block_dims, params3)
        cp.cuda.runtime.deviceSynchronize()
        d_old = out.copy()

        if data3d:
            params4 = (out, data, P1, P2, P3, lt, tau, dx, dy, dz)
        else:
            params4 = (out, data, P1, P2, lt, tau, dx, dy)
        DivProj_kernel(grid_dims, block_dims, params4)  # calculate divergence
        cp.cuda.runtime.deviceSynchronize()

        if data3d:
            params5 = (out, d_old, theta, dx, dy, dz)
        else:
            params5 = (out, d_old, theta, dx, dy)
        getU_kernel(grid_dims, block_dims, params5)
        cp.cuda.runtime.deviceSynchronize()
    return out
