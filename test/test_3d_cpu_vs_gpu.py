from numpy._typing._array_like import NDArray
import pytest
import numpy as np
from ccpi.filters.regularisers import (
    ROF_TV,
    FGP_TV,
    PD_TV,
    SB_TV,
    TGV,
    LLT_ROF,
    FGP_dTV,
    NDF,
    Diff4th,
)
from conftest import rmse, printParametersToString
from numpy.testing import assert_allclose


def test_ROF_TV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    # set parameters
    pars = {
        "algorithm": ROF_TV,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 20,
        "time_marching_parameter": 0.001,
        "tolerance_constant": 0.0,
    }
    print("#############ROF TV CPU####################")
    rof_cpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, rof_cpu)
    print("##############ROF TV GPU##################")
    rof_gpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, rof_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(rof_cpu), np.max(rof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert rof_cpu.dtype == np.float32
    assert rof_gpu.dtype == np.float32
    print("--------Results match--------")


def test_ROF_TV_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    # set parameters
    pars = {
        "algorithm": ROF_TV,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 20,
        "time_marching_parameter": 0.001,
        "tolerance_constant": 0.0,
    }
    print("#############ROF TV CPU####################")
    rof_cpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, rof_cpu)
    print("##############ROF TV GPU##################")
    rof_gpu = ROF_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, rof_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(rof_cpu), np.max(rof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert rof_cpu.dtype == np.float32
    assert rof_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def test_FGP_TV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": FGP_TV,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 30,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
    }

    print("#############FGP TV CPU####################")
    fgp_cpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, fgp_cpu)
    print("##############FGP TV GPU##################")
    fgp_gpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, fgp_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_cpu), np.max(fgp_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_cpu.dtype == np.float32
    assert fgp_gpu.dtype == np.float32
    print("--------Results match--------")


def test_FGP_TV_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": FGP_TV,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 30,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
    }

    print("#############FGP TV CPU####################")
    fgp_cpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, fgp_cpu)
    print("##############FGP TV GPU##################")
    fgp_gpu = FGP_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, fgp_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_cpu), np.max(fgp_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_cpu.dtype == np.float32
    assert fgp_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_PD_TV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": PD_TV,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
        "lipschitz_const": 8,
    }

    print("#############PD TV CPU####################")
    pd_cpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, pd_cpu)
    print("##############PD TV GPU##################")
    pd_gpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, pd_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(pd_cpu), np.max(pd_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert pd_cpu.dtype == np.float32
    assert pd_gpu.dtype == np.float32
    print("--------Results match--------")


def test_PD_TV_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": PD_TV,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "methodTV": 0,
        "nonneg": 0,
        "lipschitz_const": 8,
    }

    print("#############PD TV CPU####################")
    pd_cpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, pd_cpu)
    print("##############PD TV GPU##################")
    pd_gpu = PD_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["lipschitz_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, pd_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(pd_cpu), np.max(pd_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert pd_cpu.dtype == np.float32
    assert pd_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_SB_TV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": SB_TV,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "methodTV": 0,
    }
    print("#############SB TV CPU####################")
    sb_cpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, sb_cpu)
    print("##############SB TV GPU##################")
    sb_gpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, sb_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(sb_cpu), np.max(sb_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert sb_cpu.dtype == np.float32
    assert sb_gpu.dtype == np.float32
    print("--------Results match--------")


def test_SB_TV_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": SB_TV,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "methodTV": 0,
    }
    print("#############SB TV CPU####################")
    sb_cpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, sb_cpu)
    print("##############SB TV GPU##################")
    sb_gpu = SB_TV(
        pars["input"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["methodTV"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, sb_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(sb_cpu), np.max(sb_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert sb_cpu.dtype == np.float32
    assert sb_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# TODO: This test fails! A bug in TGV.
# def test_TGV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
#     pars = {
#         "algorithm": TGV,
#         "input": host_pepper_3d_noise,
#         "regularisation_parameter": 0.02,
#         "alpha1": 1.0,
#         "alpha0": 2.0,
#         "number_of_iterations": 50,
#         "LipshitzConstant": 12,
#         "tolerance_constant": 0.0,
#     }
#     print("#############TGV CPU####################")
#     tgv_cpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="cpu",
#     )
#     rms_cpu = rmse(host_pepper_3d, tgv_cpu)
#     print("##############TGV GPU##################")
#     tgv_gpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="gpu",
#     )
#     rms_gpu = rmse(host_pepper_3d, tgv_gpu)
#     pars["rmse"] = rms_gpu
#     txtstr = printParametersToString(pars)
#     print(txtstr)

#     print("--------Compare the results--------")
#     eps = 1e-5
#     assert_allclose(rms_cpu, rms_gpu, rtol=eps)
#     assert_allclose(np.max(tgv_cpu), np.max(tgv_gpu), rtol=eps)
#     assert rms_cpu > 0.0
#     assert rms_gpu > 0.0
#     assert tgv_cpu.dtype == np.float32
#     assert tgv_gpu.dtype == np.float32
#     print("--------Results match--------")


# TODO: This test fails! A bug in TGV.
# def test_TGV_CPU_vs_GPU_nonsquare(
#     host_pepper_im_nonsquare, host_pepper_im_noise_nonsquare
# ):
#     pars = {
#         "algorithm": TGV,
#         "input": host_pepper_im_noise_nonsquare,
#         "regularisation_parameter": 0.02,
#         "alpha1": 1.0,
#         "alpha0": 2.0,
#         "number_of_iterations": 1000,
#         "LipshitzConstant": 12,
#         "tolerance_constant": 0.0,
#     }
#     print("#############TGV CPU####################")
#     tgv_cpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="cpu",
#     )
#     rms_cpu = rmse(host_pepper_im_nonsquare, tgv_cpu)
#     print("##############TGV GPU##################")
#     tgv_gpu = TGV(
#         pars["input"],
#         pars["regularisation_parameter"],
#         pars["alpha1"],
#         pars["alpha0"],
#         pars["number_of_iterations"],
#         pars["LipshitzConstant"],
#         pars["tolerance_constant"],
#         device="gpu",
#     )
#     rms_gpu = rmse(host_pepper_im_nonsquare, tgv_gpu)

#     print("--------Compare the results--------")
#     eps = 1e-5
#     assert_allclose(rms_cpu, rms_gpu, rtol=eps)
#     assert_allclose(np.max(tgv_cpu), np.max(tgv_gpu), rtol=eps)
#     assert rms_cpu > 0.0
#     assert rms_gpu > 0.0
#     assert tgv_cpu.dtype == np.float32
#     assert tgv_gpu.dtype == np.float32


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_LLT_ROF_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": LLT_ROF,
        "input": host_pepper_3d_noise,
        "regularisation_parameterROF": 0.01,
        "regularisation_parameterLLT": 0.0085,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############LLT_ROF CPU####################")
    lltrof_cpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, lltrof_cpu)
    print("##############LLT_ROF GPU##################")
    lltrof_gpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, lltrof_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(lltrof_cpu), np.max(lltrof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert lltrof_cpu.dtype == np.float32
    assert lltrof_gpu.dtype == np.float32
    print("--------Results match--------")


def test_LLT_ROF_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": LLT_ROF,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameterROF": 0.01,
        "regularisation_parameterLLT": 0.0085,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############LLT_ROF CPU####################")
    lltrof_cpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, lltrof_cpu)
    print("##############LLT_ROF GPU##################")
    lltrof_gpu = LLT_ROF(
        pars["input"],
        pars["regularisation_parameterROF"],
        pars["regularisation_parameterLLT"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, lltrof_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(lltrof_cpu), np.max(lltrof_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert lltrof_cpu.dtype == np.float32
    assert lltrof_gpu.dtype == np.float32
    print("--------Results match--------")

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_NDF_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": NDF,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.02,
        "edge_parameter": 0.017,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.01,
        "penalty_type": 1,
        "tolerance_constant": 0.0,
    }
    print("#############NDF CPU####################")
    ndf_cpu = NDF(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["penalty_type"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, ndf_cpu)
    print("##############NDF GPU##################")
    ndf_gpu = NDF(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["penalty_type"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_gpu = rmse(host_pepper_3d, ndf_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(ndf_cpu), np.max(ndf_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert ndf_cpu.dtype == np.float32
    assert ndf_gpu.dtype == np.float32
    print("--------Results match--------")


def test_NDF_CPU_vs_GPU_noncubic(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": NDF,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.02,
        "edge_parameter": 0.017,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.01,
        "penalty_type": 1,
        "tolerance_constant": 0.0,
    }
    print("#############NDF CPU####################")
    ndf_cpu = NDF(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["penalty_type"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, ndf_cpu)
    print("##############NDF GPU##################")
    ndf_gpu = NDF(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["penalty_type"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, ndf_gpu)

    print("--------Compare the results--------")
    eps = 1e-4
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(ndf_cpu), np.max(ndf_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert ndf_cpu.dtype == np.float32
    assert ndf_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_Diff4th_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    pars = {
        "algorithm": Diff4th,
        "input": host_pepper_3d_noise,
        "regularisation_parameter": 0.8,
        "edge_parameter": 0.02,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############Diff4th CPU####################")
    diff4th_cpu = Diff4th(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, diff4th_cpu)
    print("##############Diff4th GPU##################")
    diff4th_gpu = Diff4th(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, diff4th_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-4
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(diff4th_cpu), np.max(diff4th_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert diff4th_cpu.dtype == np.float32
    assert diff4th_gpu.dtype == np.float32
    print("--------Results match--------")


def test_Diff4th_CPU_vs_GPU_nonsquare(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    pars = {
        "algorithm": Diff4th,
        "input": host_pepper_3d_noise_noncubic,
        "regularisation_parameter": 0.8,
        "edge_parameter": 0.02,
        "number_of_iterations": 50,
        "time_marching_parameter": 0.0001,
        "tolerance_constant": 0.0,
    }
    print("#############Diff4th CPU####################")
    diff4th_cpu = Diff4th(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, diff4th_cpu)
    print("##############Diff4th GPU##################")
    diff4th_gpu = Diff4th(
        pars["input"],
        pars["regularisation_parameter"],
        pars["edge_parameter"],
        pars["number_of_iterations"],
        pars["time_marching_parameter"],
        pars["tolerance_constant"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, diff4th_gpu)

    print("--------Compare the results--------")
    eps = 1e-4
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(diff4th_cpu), np.max(diff4th_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert diff4th_cpu.dtype == np.float32
    assert diff4th_gpu.dtype == np.float32
    print("--------Results match--------")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


def test_FGP_dTV_CPU_vs_GPU(host_pepper_3d, host_pepper_3d_noise):
    # set parameters
    pars = {
        "algorithm": FGP_dTV,
        "input": host_pepper_3d_noise,
        "refdata": host_pepper_3d,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "eta_const": 0.2,
        "methodTV": 0,
        "nonneg": 0,
    }
    print("#############FGP_dTV CPU####################")
    fgp_dtv_cpu = FGP_dTV(
        pars["input"],
        pars["refdata"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["eta_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d, fgp_dtv_cpu)
    print("##############FGP_dTV GPU##################")
    fgp_dtv_gpu = FGP_dTV(
        pars["input"],
        pars["refdata"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["eta_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d, fgp_dtv_gpu)
    pars["rmse"] = rms_gpu
    txtstr = printParametersToString(pars)
    print(txtstr)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_dtv_cpu), np.max(fgp_dtv_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_dtv_cpu.dtype == np.float32
    assert fgp_dtv_gpu.dtype == np.float32
    print("--------Results match--------")


def test_FGP_dTV_CPU_vs_GPU_nonsquare(
    host_pepper_3d_noncubic, host_pepper_3d_noise_noncubic
):
    # set parameters
    pars = {
        "algorithm": FGP_dTV,
        "input": host_pepper_3d_noise_noncubic,
        "refdata": host_pepper_3d_noncubic,
        "regularisation_parameter": 0.02,
        "number_of_iterations": 50,
        "tolerance_constant": 0.0,
        "eta_const": 0.2,
        "methodTV": 0,
        "nonneg": 0,
    }
    print("#############FGP_dTV CPU####################")
    fgp_dtv_cpu = FGP_dTV(
        pars["input"],
        pars["refdata"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["eta_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="cpu",
    )
    rms_cpu = rmse(host_pepper_3d_noncubic, fgp_dtv_cpu)
    print("##############FGP_dTV GPU##################")
    fgp_dtv_gpu = FGP_dTV(
        pars["input"],
        pars["refdata"],
        pars["regularisation_parameter"],
        pars["number_of_iterations"],
        pars["tolerance_constant"],
        pars["eta_const"],
        pars["methodTV"],
        pars["nonneg"],
        device="gpu",
    )
    rms_gpu = rmse(host_pepper_3d_noncubic, fgp_dtv_gpu)

    print("--------Compare the results--------")
    eps = 1e-5
    assert_allclose(rms_cpu, rms_gpu, rtol=eps)
    assert_allclose(np.max(fgp_dtv_cpu), np.max(fgp_dtv_gpu), rtol=eps)
    assert rms_cpu > 0.0
    assert rms_gpu > 0.0
    assert fgp_dtv_cpu.dtype == np.float32
    assert fgp_dtv_gpu.dtype == np.float32
    print("--------Results match--------")
