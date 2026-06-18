@echo off
setlocal
rem ============================================================
rem  AI Olympics 2026 - Quick Launcher  (team Ded_Sec)
rem
rem  Double-click           -> menu, type 1-8 / v / p
rem  run.bat 3              -> run Game 3 (SAFE: writes to _retest\game3)
rem  run.bat 3 live         -> run Game 3 into the LIVE submission folder
rem  run.bat v              -> validate all 8 submission packages
rem  run.bat p "C:\x.csv"   -> Phase 2 runner on that test CSV
rem ============================================================

set "PROJECT=C:\Users\mohd1\OneDrive\Desktop\ai olpyc"
if not exist "%PROJECT%\Game 1.py" set "PROJECT=%~dp0"
cd /d "%PROJECT%" || goto :fail

set "CHOICE=%~1"
set "ARG2=%~2"
if not "%CHOICE%"=="" goto :dispatch

echo.
echo  ==============================================
echo    AI Olympics 2026 - Quick Launcher (Ded_Sec)
echo  ==============================================
echo    1-8 : run that game
echo    v   : validate all 8 submissions
echo    p   : Phase 2 runner (final test, 16 June)
echo  ==============================================
set /p CHOICE=  Enter 1-8, v or p:

:dispatch
if /i "%CHOICE%"=="v" goto :validate
if /i "%CHOICE%"=="p" goto :phase2
if exist "Game %CHOICE%.py" goto :game
echo.
echo  [ERROR] "Game %CHOICE%.py" not found. Valid choices: 1-8, v, p.
goto :done

:game
set "MODE=safe"
if /i "%ARG2%"=="live" set "MODE=live"
if not "%~1"=="" goto :game_run
set /p LIVEANS=  Overwrite the LIVE submission folder? (y/N):
if /i "%LIVEANS%"=="y" set "MODE=live"

:game_run
echo.
if "%MODE%"=="live" (
    echo  [WARNING] Writing into the LIVE submission folder.
    echo  After it finishes, re-execute the notebook or "run.bat v" will FAIL.
    echo.
    python "Game %CHOICE%.py"
) else (
    echo  SAFE mode: output goes to "_retest\game%CHOICE%" - submission untouched.
    echo.
    python "Game %CHOICE%.py" --output-dir "_retest\game%CHOICE%"
)
goto :done

:validate
cd /d "%PROJECT%\Tools Used To Find Results"
python validate_submissions.py
goto :done

:phase2
if not "%ARG2%"=="" set "CSV=%ARG2%"
if "%ARG2%"=="" set /p CSV=  Full path to the test CSV:
if not defined CSV (
    echo  [ERROR] No CSV path given.
    goto :done
)
set "CSV=%CSV:"=%"
if not exist "%CSV%" (
    echo  [ERROR] File not found: "%CSV%"
    goto :done
)
cd /d "%PROJECT%\Tools Used To Find Results"
python phase2_runner.py --test-csv "%CSV%"
echo.
echo  Output: phase2_predictions.csv  (in "Tools Used To Find Results")
goto :done

:fail
echo  [ERROR] Project folder not found: "%PROJECT%"
pause
exit /b 1

:done
echo.
pause
endlocal
