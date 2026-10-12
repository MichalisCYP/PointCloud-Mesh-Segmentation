@echo off
rem Segment point clouds and meshes on Windows. Usage:
rem   segment.bat my_scan.las            one file
rem   segment.bat D:\scans               every .ply .las .laz .txt .obj .glb .fbx in a folder
rem Results go to results\<name>\  (open the .ply files in CloudCompare or MeshLab)
rem Extra options are passed through, e.g.  segment.bat facade.obj --up y
setlocal enabledelayedexpansion
set "REPO=%~dp0"
set "PY=%REPO%.venv\Scripts\python.exe"
if not exist "%PY%" ( echo Run install.bat first. & exit /b 1 )
if "%~1"=="" ( echo Usage: segment.bat ^<file or folder^> [options] & exit /b 1 )

set "IN=%~1"
shift
set "EXTRA="
:collect
if "%~1"=="" goto run
set "EXTRA=!EXTRA! %1"
shift
goto collect

:run
if exist "%IN%\*" (
  for %%E in (ply las laz txt obj glb gltf fbx) do (
    for %%F in ("%IN%\*.%%E") do (
      echo %%~nF | findstr /i /c:".converted" >nul || call :one "%%~fF"
    )
  )
) else (
  call :one "%IN%"
)
echo Done. Open results\^<name^>\points_labelled.ply (and mesh_labelled.ply for meshes).
exit /b 0

:one
echo === %~n1 ===
"%PY%" "%REPO%scripts\zero_shot_segment.py" "%~1" "%REPO%results\%~n1" --views both --save-views %EXTRA%
exit /b 0
