param(
    [ValidateSet("check", "trace", "smoke", "train", "eval-base", "eval-qlora", "eval-challenge", "analyze", "benchmark", "demo")]
    [string]$Mode = "check"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = "C:\anaconda\envs\gemma-workorder\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "未找到 gemma-workorder 环境：$Python"
}

Set-Location -LiteralPath $ProjectRoot

function Invoke-ProjectPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "命令执行失败，退出码：$LASTEXITCODE"
    }
}

switch ($Mode) {
    "check" {
        Invoke-ProjectPython -Arguments @(
            "-c",
            "import torch; print('Python environment ready'); print('torch =', torch.__version__); print('CUDA =', torch.cuda.is_available()); print('GPU =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
        )
        Invoke-ProjectPython -Arguments @("-m", "pytest", "-q")
    }
    "trace" {
        Invoke-ProjectPython -Arguments @(
            "scripts/export_agent_traces.py",
            "--input", "data/tool_use/train.jsonl",
            "--output", "artifacts/traces/train.jsonl"
        )
    }
    "smoke" {
        Invoke-ProjectPython -Arguments @(
            "scripts/train_tool_use_qlora.py",
            "--max-train-samples", "32",
            "--max-validation-samples", "32",
            "--epochs", "1",
            "--warmup-steps", "1",
            "--output-dir", "artifacts/tool_use_smoke_adapter"
        )
    }
    "train" {
        Invoke-ProjectPython -Arguments @(
            "scripts/train_tool_use_qlora.py",
            "--output-dir", "artifacts/tool_use_qlora_adapter"
        )
    }
    "eval-base" {
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--precision", "4bit",
            "--output", "reports/tool_use_base_4bit.json"
        )
    }
    "eval-qlora" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "尚未找到正式 Adapter，请先运行：.\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--precision", "4bit",
            "--output", "reports/tool_use_qlora_4bit.json"
        )
    }
    "eval-challenge" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "尚未找到正式 Adapter，请先运行：.\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/evaluate_tool_use.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--precision", "4bit",
            "--dataset", "data/tool_use/challenge.jsonl",
            "--output", "reports/tool_use_qlora_challenge_4bit.json"
        )
    }
    "analyze" {
        Invoke-ProjectPython -Arguments @("scripts/analyze_tool_use_results.py")
    }
    "benchmark" {
        Invoke-ProjectPython -Arguments @("scripts/run_agent_benchmark.py")
    }
    "demo" {
        if (-not (Test-Path -LiteralPath "artifacts/tool_use_qlora_adapter")) {
            throw "尚未找到正式 Adapter，请先运行：.\run_local.ps1 -Mode train"
        }
        Invoke-ProjectPython -Arguments @(
            "scripts/run_tool_use_demo.py",
            "--adapter", "artifacts/tool_use_qlora_adapter",
            "--index", "0"
        )
    }
}
