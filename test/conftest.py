# Defines common fixtures and makes them available to all tests

import os
import subprocess
from imageio.v2 import imread

import numpy as np
import pytest

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

# nvidia
try:
    subprocess.check_output("nvidia-smi")
    has_nvidia = True
except:
    has_nvidia = False

CUR_DIR = os.path.abspath(os.path.dirname(__file__))


def pytest_addoption(parser):
    parser.addoption(
        "--runcupy", action="store_true", default=False, help="run cupy tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runcupy"):
        # --runcupy given in cli: do not skip cupy tests
        return
    skip_cupy = pytest.mark.skip(reason="need --runcupy option to run")
    for item in items:
        if "cupy" in item.keywords:
            item.add_marker(skip_cupy)


@pytest.fixture(scope="function", autouse=True)
def my_common_fixture(request):
    if not has_nvidia:
        pytest.skip("GPU is not available, the test is skipped")


@pytest.fixture(scope="session")
def test_data_path():
    return os.path.join(CUR_DIR, "test_data")


# only load from disk once per session, and we use np.copy for the elements,
# to ensure data in this loaded file stays as originally loaded
@pytest.fixture(scope="session")
def data_file(test_data_path):
    in_file = os.path.join(test_data_path, "tomo_standard.npz")
    return np.load(in_file)


@pytest.fixture(scope="session")
def host_pepper_im(test_data_path):
    in_file = os.path.join(test_data_path, "peppers.tif")
    Im = imread(in_file)
    Im = Im / 255
    return Im.astype("float32")


@pytest.fixture(scope="session")
def host_pepper_im_nonsquare(test_data_path):
    in_file = os.path.join(test_data_path, "peppers.tif")
    Im = imread(in_file)
    Im_cropped = Im[0:415, 0:460]
    Im_cropped = Im_cropped / 255
    return Im_cropped.astype("float32")


@pytest.fixture(scope="session")
def host_pepper_im_noise(host_pepper_im):
    perc = 0.05
    u0 = host_pepper_im + np.random.normal(
        loc=0, scale=perc * host_pepper_im, size=np.shape(host_pepper_im)
    )
    u0 = u0.astype("float32")
    return u0


@pytest.fixture(scope="session")
def host_pepper_3d(host_pepper_im):
    slices_no = 5
    (x_size, y_size) = np.shape(host_pepper_im)
    GT_vol = np.zeros((slices_no, x_size, y_size), dtype="float32")
    for i in range(slices_no):
        GT_vol[i, :, :] = host_pepper_im
    return GT_vol


@pytest.fixture(scope="session")
def host_pepper_3d_noise(host_pepper_3d):
    perc = 0.075
    u0 = host_pepper_3d + np.random.normal(
        loc=0, scale=perc * host_pepper_3d, size=np.shape(host_pepper_3d)
    )
    u0 = u0.astype("float32")
    return u0


@pytest.fixture(scope="session")
def host_pepper_im_noise_nonsquare(host_pepper_im_nonsquare):
    perc = 0.05
    u0 = host_pepper_im_nonsquare + np.random.normal(
        loc=0,
        scale=perc * host_pepper_im_nonsquare,
        size=np.shape(host_pepper_im_nonsquare),
    )
    u0 = u0.astype("float32")
    return u0


@pytest.fixture(scope="session")
def host_pepper_3d_noncubic(host_pepper_im_nonsquare):
    slices_no = 5
    (x_size, y_size) = np.shape(host_pepper_im_nonsquare)
    GT_vol = np.zeros((slices_no, x_size, y_size), dtype="float32")
    for i in range(slices_no):
        GT_vol[i, :, :] = host_pepper_im_nonsquare
    return GT_vol


@pytest.fixture(scope="session")
def host_pepper_3d_noise_noncubic(host_pepper_3d_noncubic):
    perc = 0.075
    u0 = host_pepper_3d_noncubic + np.random.normal(
        loc=0,
        scale=perc * host_pepper_3d_noncubic,
        size=np.shape(host_pepper_3d_noncubic),
    )
    u0 = u0.astype("float32")
    return u0


@pytest.fixture
def device_pepper_im(host_pepper_im, ensure_clean_memory):
    return xp.ascontiguousarray(xp.asarray(host_pepper_im, order="C"), dtype=np.float32)


@pytest.fixture
def device_pepper_im_noise(host_pepper_im_noise, ensure_clean_memory):
    return xp.ascontiguousarray(
        xp.asarray(host_pepper_im_noise, order="C"), dtype=np.float32
    )


@pytest.fixture
def ensure_clean_memory():
    xp.get_default_memory_pool().free_all_blocks()
    xp.get_default_pinned_memory_pool().free_all_blocks()
    yield None
    xp.get_default_memory_pool().free_all_blocks()
    xp.get_default_pinned_memory_pool().free_all_blocks()


@pytest.fixture
def host_data(data_file):
    return np.copy(data_file["data"])


@pytest.fixture
def device_data(host_data, ensure_clean_memory):
    return xp.ascontiguousarray(xp.asarray(host_data, order="C"), dtype=np.float32)


def printParametersToString(pars):
    txt = r""
    for key, value in pars.items():
        if key == "algorithm":
            txt += "{0} = {1}".format(key, value.__name__)
        elif key == "input":
            txt += "{0} = {1}".format(key, np.shape(value))
        elif key == "refdata":
            txt += "{0} = {1}".format(key, np.shape(value))
        else:
            txt += "{0} = {1}".format(key, value)
        txt += "\n"
    return txt


def nrmse(im1, im2):
    rmse = np.sqrt(np.sum((im2 - im1) ** 2) / float(im1.size))
    max_val = max(np.max(im1), np.max(im2))
    min_val = min(np.min(im1), np.min(im2))
    return 1 - (rmse / (max_val - min_val))


def rmse(im1, im2):
    rmse = np.sqrt(np.sum((im1 - im2) ** 2) / float(im1.size))
    return rmse
