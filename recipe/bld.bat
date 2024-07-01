mkdir "%SRC_DIR%\test"
ROBOCOPY /E "%RECIPE_DIR%\..\test" "%SRC_DIR%\test"
if exist "%RECIPE_DIR%\..\build_proj" (
    rd /s /q "%RECIPE_DIR%\..\build_proj"
)

mkdir "%SP_DIR%\ccpi\cuda_kernels"
ROBOCOPY /E "%RECIPE_DIR%\..\src\Core\regularisers_GPU\cuda_kernels\" "%SP_DIR%\ccpi\"

:: add -G "Visual Studio 16 2019" to the cmake command to specify the generator
:: add -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4" to the cmake command to specify the CUDA toolkit
cmake -S "%SRC_DIR%" -B "%RECIPE_DIR%\..\build_proj" -DBUILD_PYTHON_WRAPPERS=ON -DCONDA_BUILD=ON -DBUILD_CUDA=ON -DCMAKE_BUILD_TYPE="RelWithDebInfo" -DLIBRARY_INC="%CONDA_PREFIX%" -DCMAKE_INSTALL_PREFIX="%RECIPE_DIR%\..\install"
cmake --build "%RECIPE_DIR%\..\build_proj" --target install --config RelWithDebInfo
%PYTHON% -m pip install "%SRC_DIR%\src\Python"

if errorlevel 1 exit 1
