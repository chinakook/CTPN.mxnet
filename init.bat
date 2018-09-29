mkdir .\model
call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" x64
set PATH=C:\Program Files (x86)\Windows Kits\10\bin\10.0.17134.0\x64;%PATH%
cd /d %~dp0
cd rcnn\cython
python setup_windows.py build_ext --inplace
python setup_windows_cuda.py build_ext --inplace
cd ..\..
pause
