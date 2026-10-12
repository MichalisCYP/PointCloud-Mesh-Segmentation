@echo off
rem One-time Windows install: Python 3.12 venv in .venv with CUDA PyTorch + SAM 3.
rem Needs: NVIDIA GPU + driver, git, internet. Blender (optional) for FBX input.
rem Before the first run: request access at https://huggingface.co/facebook/sam3, then log in
rem with:  .venv\Scripts\hf.exe auth login
setlocal
set "REPO=%~dp0"
cd /d "%REPO%"

where uv >nul 2>nul || python -m pip install --user -q uv
for /f "delims=" %%U in ('python -c "import uv; print(uv.find_uv_bin())"') do set "UV=%%U"
if not defined UV set "UV=uv"

if not exist .venv "%UV%" venv .venv --python 3.12 || exit /b 1
"%UV%" pip install --python .venv\Scripts\python.exe torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu126 || exit /b 1
if not exist "%REPO%..\sam3" git clone https://github.com/facebookresearch/sam3.git "%REPO%..\sam3" || exit /b 1
"%UV%" pip install --python .venv\Scripts\python.exe -e "%REPO%..\sam3" "setuptools<81" einops pycocotools psutil ^
  scipy plyfile trimesh "laspy[lazrs]" pillow huggingface_hub "triton-windows<3.7" || exit /b 1

echo.
echo Installed. Next:
echo   1. .venv\Scripts\hf.exe auth login      (once, after your SAM 3 access is approved)
echo   2. segment.bat ^<file or folder^>
