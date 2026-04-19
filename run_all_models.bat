@echo off
echo =======================================================
echo Running all 4 models
echo =======================================================
echo.

echo [1/4] Running PhoBERT...
python src\models\train_phobert.py
taskkill /f /im python.exe /fi "memusage gt 1" >nul 2>&1
if errorlevel 1 goto error

echo.
echo [2/4] Running BART-pho...
python src\models\train_bartpho.py
taskkill /f /im python.exe /fi "memusage gt 1" >nul 2>&1
if errorlevel 1 goto error

echo.
echo [3/4] Running XLM-RoBERTa...
python src\models\train_xlmroberta.py
if errorlevel 1 goto error

echo.
echo [4/4] Running mBERT...
python src\models\train_mbert.py
if errorlevel 1 goto error

echo.
echo =======================================================
echo ALL MODELS COMPLETED!
echo Results: data\processed\benchmark_results.csv
echo =======================================================
goto end

:error
echo.
echo =======================================================
echo ERROR OCCURRED! PLEASE CHECK THE PROCESS.
echo =======================================================

:end
pause
