@echo off
REM StockMate LLM server - Qwen3.6-35B-A3B with MoE CPU offloading
REM
REM Measured on RTX 4060 Ti 8GB + DDR5-5600 64GB:
REM   --parallel 1 --load-mode none : 31.4 -> 41.9 tok/s
REM   --n-cpu-moe 34 is optimal. Below 28 causes VRAM spill (speed halves).
REM
REM API key is read from .env - no secrets in this file.
REM Bound to 127.0.0.1 only. For team access, expose FastAPI (8000) instead.

set MODEL=%USERPROFILE%\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

set APIKEY=
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0.env") do if "%%a"=="LLM_API_KEY" set APIKEY=%%b
if "%APIKEY%"=="" (
  echo [ERROR] LLM_API_KEY not found in .env
  pause
  exit /b 1
)

cd /d "C:\Users\dhp13\Downloads\llama"
llama-server.exe -m "%MODEL%" -ngl 999 --n-cpu-moe 34 -fa on -c 16384 --parallel 1 --load-mode none --jinja --alias qwen3.6-35b --host 127.0.0.1 --port 8080 --api-key %APIKEY%
