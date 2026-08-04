@echo off
REM ============================================================
REM  StockMate LLM ????  (Qwen3.6-35B-A3B, MoE CPU ?????ех?)
REM ------------------------------------------------------------
REM  ???? ??? (RTX 4060 Ti 8GB + DDR5-5600 64GB)
REM    --parallel 1 --no-mmap : 31.4 -> 41.9 tok/s
REM    --n-cpu-moe 34         : ????. 28 ????? VRAM ????? ????
REM
REM  API ??? .env ???? ?м╒ве?. ?? ??????? ??м╤??? ???? ??ве?.
REM  127.0.0.1 ???ех? = ?? PC ????. ???? ?????? FastAPI(8000)?? ????.
REM ============================================================

set MODEL=%USERPROFILE%\.cache\huggingface\hub\models--unsloth--Qwen3.6-35B-A3B-GGUF\snapshots\a483e9e6cbd595906af30beda3187c2663a1118c\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0.env") do (
  if "%%a"=="LLM_API_KEY" set APIKEY=%%b
)
if "%APIKEY%"=="" (
  echo [!] .env ???? LLM_API_KEY ?? a?? ????????.
  pause
  exit /b 1
)

cd /d "C:\Users\dhp13\Downloads\llama"
llama-server.exe -m "%MODEL%" ^
  -ngl 999 --n-cpu-moe 34 ^
  -fa on -c 16384 ^
  --parallel 1 --no-mmap ^
  --jinja --alias qwen3.6-35b ^
  --host 127.0.0.1 --port 8080 ^
  --api-key %APIKEY%
