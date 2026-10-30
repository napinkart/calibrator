@echo off
chcp 65001 >nul
title LibreWarp
cd /d "%~dp0"

rem ---- detectar entorno virtual funcional ----
rem Prioridad: 1. .venv local (creado por este mismo bat), 2. system python,
rem 3. crear .venv desde cero. NO se prioriza ..\.venv-sim: es un venv grande
rem y portátil, y arrastrarlo al copiar el repo en otra PC rompe la instalación.
set PY=
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import cv2" >nul 2>nul
    if not errorlevel 1 set PY=".venv\Scripts\python.exe"
)
if not defined PY (
    python -c "import cv2" >nul 2>nul
    if not errorlevel 1 set PY=python
)
if not defined PY (
    echo [INFO] Creando o reparando entorno virtual .venv...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if not exist ".venv\Scripts\python.exe" (
        echo.
        echo [ERROR] No se encontro Python funcional. Instala Python 3.10-3.12 o 3.13 desde python.org
        pause
        exit /b 1
    )
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    set PY=".venv\Scripts\python.exe"
)

if not exist "config.json" (
    copy config.example.json config.json >nul
    echo Se creo config.json a partir de config.example.json
)
rem ---- soporte NDI (camara del telefono): se instala si falta ----
%PY% -c "import cyndilib" >nul 2>nul
if errorlevel 1 (
    echo Instalando soporte NDI [cyndilib]...
    %PY% -m pip install cyndilib
)

:menu
cls
echo ============================================================
echo                 LibreWarp - calibrador
echo ============================================================
echo   0. Prueba simulada (sin hardware, verifica la instalacion)
echo   C. Prueba simulada eligiendo escena (ciclorama convexo / pantalla concaava)
echo   1. Listar monitores (para editar config.json)
echo   N. Buscar camaras NDI (telefono con app NDI)
echo   2. Editar config.json
echo   3. Identificar proyectores (verificar QuadHead2Go)
echo   4. Ajustar camara
echo   5. Escanear proyectores
echo   6. Marcar bordes de la pantalla
echo   7. Calcular mapas
echo   D. Diagnosticar el ultimo escaneo (si falla el paso 7)
echo   L. Limpiar escaneos viejos (borrar y empezar de cero)
echo   8. Verificar en los proyectores
echo   9. Abrir carpeta output (para TouchDesigner)
echo   Q. Salir
echo ============================================================
set "op="
set /p op=Opcion: 
if /i "%op%"=="0" %PY% calibrate.py simulate & pause & goto menu
if /i "%op%"=="c" goto sim_scene
if /i "%op%"=="1" %PY% calibrate.py monitors & pause & goto menu
if /i "%op%"=="n" %PY% calibrate.py ndi -c config.json & pause & goto menu
if /i "%op%"=="2" start "" notepad config.json & goto menu
if /i "%op%"=="3" %PY% calibrate.py identify -c config.json & pause & goto menu
if /i "%op%"=="4" %PY% calibrate.py adjust -c config.json & pause & goto menu
if /i "%op%"=="5" %PY% calibrate.py scan -c config.json & pause & goto menu
if /i "%op%"=="6" %PY% calibrate.py screen -c config.json & pause & goto menu
if /i "%op%"=="7" %PY% calibrate.py compute -c config.json & pause & goto menu
if /i "%op%"=="d" %PY% calibrate.py diagnose -c config.json & (if exist work\diagnostico.png start "" "work\diagnostico.png") & pause & goto menu
if /i "%op%"=="8" %PY% calibrate.py verify -c config.json & pause & goto menu
if /i "%op%"=="9" (if not exist output mkdir output) & start "" explorer "%~dp0output" & goto menu
if /i "%op%"=="l" %PY% calibrate.py clean -c config.json & goto confirm_clean
if /i "%op%"=="q" exit /b 0
goto menu

:sim_scene
cls
echo ============================================================
echo          LibreWarp - prueba simulada (sin hardware)
echo ============================================================
echo   1. Ciclorama convexo 120 grados (camara FUERA del cilindro)
echo   2. Pantalla concaava  72 grados  (camara DENTRO del cilindro)
echo   3. Pantalla concaava  90 grados  (prueba rapida, 2 proyectores)
echo   Q. Volver al menu
echo ============================================================
set "sc="
set /p sc=Elige 1, 2, 3 o Q: 
if "%sc%"=="1" goto sim_convex
if "%sc%"=="2" goto sim_concave
if "%sc%"=="3" goto sim_concave2
if /i "%sc%"=="q" goto menu
echo Opcion no valida.
pause
goto sim_scene

:sim_convex
%PY% calibrate.py simulate --scene convex3
pause
goto menu

:sim_concave
%PY% calibrate.py simulate --scene concave3
pause
goto menu

:sim_concave2
%PY% calibrate.py simulate --scene concave2
pause
goto menu

:confirm_clean
set "ok="
set /p ok=Borrar de verdad? (s/N): 
if /i "%ok%"=="s" %PY% calibrate.py clean -c config.json --yes
pause
goto menu
