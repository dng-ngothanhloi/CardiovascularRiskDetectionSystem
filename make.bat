@echo off
REM Windows batch equivalent of Makefile (key pipeline targets)
REM Usage: make.bat [command]
REM Prefer WSL + make for full parity; this covers the main stages.

setlocal EnableExtensions
set CONFIG=configs/default.yaml
set L3_CONFIG=configs/l3_discovery.yaml
set DATA=data\cardio_data.csv
set MODEL_ROOT=artifacts\Model
set EVAL_DIR=artifacts\Eval
set INTERPRET_DIR=artifacts\Interpretation

if "%1"=="" goto :help
if "%1"=="help" goto :help

if "%1"=="install" (
    echo Installing dependencies...
    pip install -r requirements.txt
    goto :end
)

if "%1"=="check-data" (
    if exist "%DATA%" ( echo Data OK: %DATA% ) else ( echo Error: %DATA% not found! & exit /b 1 )
    goto :end
)

if "%1"=="check-config" (
    if exist "%CONFIG%" ( echo Config OK: %CONFIG% ) else ( echo Error: %CONFIG% not found! & exit /b 1 )
    goto :end
)

if "%1"=="setup" (
    call make.bat install
    if errorlevel 1 goto :end
    call make.bat check-data
    if errorlevel 1 goto :end
    call make.bat check-config
    goto :end
)

if "%1"=="eda" (
    if not exist "artifacts\eda" mkdir artifacts\eda
    python -m src.data_understanding --config %CONFIG%
    goto :end
)

if "%1"=="preprocess" (
    if not exist "artifacts\preprocessing" mkdir artifacts\preprocessing
    python -m src.preprocessing --config %CONFIG%
    goto :end
)

if "%1"=="fe-rf" (
    python -m src.feature_engineering --config %CONFIG% --model-type rf
    goto :end
)

if "%1"=="fe-tabnet" (
    python -m src.feature_engineering --config %CONFIG% --model-type tabnet
    goto :end
)

if "%1"=="fe" (
    call make.bat fe-rf
    if errorlevel 1 goto :end
    call make.bat fe-tabnet
    goto :end
)

if "%1"=="train-rf" (
    if not exist "%MODEL_ROOT%" mkdir %MODEL_ROOT%
    python -m src.modeling --config %CONFIG% --model rf
    goto :end
)

if "%1"=="train-tabnet" (
    if not exist "%MODEL_ROOT%" mkdir %MODEL_ROOT%
    python -m src.modeling --config %CONFIG% --model tabnet
    goto :end
)

if "%1"=="train-vae-tabnet" (
    if not exist "%MODEL_ROOT%" mkdir %MODEL_ROOT%
    python -m src.modeling --config %CONFIG% --model vae_tabnet --joint
    goto :end
)

if "%1"=="train" (
    call make.bat train-rf
    if errorlevel 1 goto :end
    call make.bat train-tabnet
    goto :end
)

if "%1"=="evaluate" (
    if not exist "%EVAL_DIR%" mkdir %EVAL_DIR%
    python -m src.evaluation --model_root artifacts/Model --out_dir artifacts/Eval --n_boot 2000 --seed 42
    goto :end
)

if "%1"=="attention" (
    if not exist "%INTERPRET_DIR%" mkdir %INTERPRET_DIR%
    python -m src.interpretation.run_attention_summary --fe_dir_tabnet artifacts/fe/tabnet --model_root artifacts/Model/tabnet --out_root artifacts/Interpretation --rf_model_root artifacts/Model/rf --fe_dir_rf artifacts/fe/random_forest
    goto :end
)

if "%1"=="calibrate" (
    if not exist "%INTERPRET_DIR%\calibration" mkdir %INTERPRET_DIR%\calibration
    python -m src.interpretation.run_calibration --model rf --method platt --model_root artifacts/Model --out_root artifacts/Interpretation/calibration --n_bins 15
    if errorlevel 1 goto :end
    python -m src.interpretation.run_calibration --model tabnet --method isotonic --model_root artifacts/Model --out_root artifacts/Interpretation/calibration --n_bins 15
    goto :end
)

if "%1"=="lifestyle-prep" (
    if not exist "%INTERPRET_DIR%\test_prediction_result" mkdir %INTERPRET_DIR%\test_prediction_result
    python -m src.test_prediction_verification --n_samples 0 --test_prediction src/test_prediction_input.txt
    if errorlevel 1 goto :end
    python -m src.create_lifestyle_enriched_predictions --fe_dir artifacts/fe --predictions_dir artifacts/Interpretation/test_prediction_result --output_dir artifacts/Interpretation/test_prediction_result
    goto :end
)

if "%1"=="xai" (
    call make.bat lifestyle-prep
    if errorlevel 1 goto :end
    if not exist "%INTERPRET_DIR%\reports" mkdir %INTERPRET_DIR%\reports
    python -m src.xai_interpretability --output-dir artifacts/Interpretation/reports
    goto :end
)

if "%1"=="interpret" (
    call make.bat attention
    if errorlevel 1 goto :end
    call make.bat calibrate
    if errorlevel 1 goto :end
    call make.bat xai
    goto :end
)

if "%1"=="discover" (
    if not exist "artifacts\Discovery" mkdir artifacts\Discovery
    python -m src.run_discovery --config %L3_CONFIG%
    goto :end
)

if "%1"=="discover-fast" (
    if not exist "artifacts\Discovery" mkdir artifacts\Discovery
    python -m src.run_discovery --config %L3_CONFIG% --tau 0.5 --min_support 0.02 --min_conf 0.6 --min_lift 1.2 --max_len 3 --no_bootstrap
    goto :end
)

if "%1"=="pipeline-l1" (
    call make.bat eda
    if errorlevel 1 goto :end
    call make.bat preprocess
    if errorlevel 1 goto :end
    call make.bat fe
    if errorlevel 1 goto :end
    call make.bat train
    if errorlevel 1 goto :end
    call make.bat evaluate
    goto :end
)

if "%1"=="all" (
    call make.bat pipeline-l1
    if errorlevel 1 goto :end
    call make.bat interpret
    if errorlevel 1 goto :end
    call make.bat discover
    echo Full pipeline completed.
    goto :end
)

if "%1"=="clean" (
    if exist "artifacts" rmdir /s /q artifacts
    if exist "reports" rmdir /s /q reports
    echo Clean complete!
    goto :end
)

echo Unknown command: %1
echo Use: make.bat help
exit /b 1

:help
echo CardiovascularRiskDetectionSystem - Windows Batch Commands
echo.
echo Usage: make.bat [command]
echo.
echo Setup:     setup ^| install ^| check-data ^| check-config
echo Stages:    eda ^| preprocess ^| fe ^| train ^| evaluate ^| interpret ^| discover
echo Train:     train-rf ^| train-tabnet ^| train-vae-tabnet
echo Full:      pipeline-l1 ^| all
echo Other:     discover-fast ^| clean ^| attention ^| calibrate ^| xai
echo.
echo Data path: %DATA%
echo See README.md and docs/Flow-Pipeline.md
goto :end

:end
endlocal
