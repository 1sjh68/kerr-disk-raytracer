# run_ncu_pipeline.ps1
# 一键运行 Windows host Nsight Compute 完整 profiling 流水线
#
# 用法（普通 PowerShell 即可）：
#   powershell -File 'D:\Desktop\black hole\tools\run_ncu_pipeline.ps1'
#
# 流程：
#   1. 弹一次 UAC -> 你点 "是"
#   2. 提权 PowerShell 跑 _ncu_full_pipeline.ps1
#   3. 该脚本对项目跑 4 个 ncu --set full：
#        float64 @ 48, float64 @ 128, float32 @ 48, float32 @ 128
#   4. 产物落入 D:\Desktop\black hole\results\
#        - ncu_*.ncu-rep      .ncu-rep 二进制（Nsight Compute UI 可打开）
#        - ncu_*.summary.txt  per-kernel 文本摘要
#        - ncu_*.csv          完整 metrics CSV
#        - ncu_*.run.log      ncu 运行日志
#
# 整体耗时 5-15 分钟，期间不要做其他 GPU 重活，让 ncu 独占 GPU 拿稳定数据。

$ErrorActionPreference = 'Continue'
$logFile  = "$env:TEMP\ncu_full_pipeline.log"
$sentinel = "$env:TEMP\ncu_pipeline_done.txt"
$script   = 'D:\Desktop\black hole\tools\_ncu_full_pipeline.ps1'

# 清旧
Get-Item $sentinel, "$sentinel.start", $logFile -ErrorAction SilentlyContinue | Remove-Item -Force

if (-not (Test-Path $script)) {
    Write-Host "❌ 找不到 $script" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat')) {
    Write-Host "❌ 找不到 Nsight Compute 2024.3.2 安装" -ForegroundColor Red
    Write-Host "   预期路径：C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\"
    exit 1
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Nsight Compute Full Profiling Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "屏幕马上会变暗 + 弹 UAC 对话框（'是否要允许此应用对你的设备进行更改'）"
Write-Host "请点 [是]"
Write-Host ""
Write-Host "之后这里会显示进度，约 5-15 分钟"
Write-Host ""
Write-Host "按任意键继续..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')

$proc = Start-Process powershell -Verb RunAs -PassThru `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, `
                  "-LogFile", $logFile, "-Sentinel", $sentinel

if ($null -eq $proc) {
    Write-Host ""
    Write-Host "❌ UAC 被取消" -ForegroundColor Red
    Write-Host "   重新运行本脚本即可"
    exit 1
}

Write-Host ""
Write-Host "✅ 提权进程启动 (PID = $($proc.Id))，开始 poll..." -ForegroundColor Green
Write-Host ""

$start = Get-Date
$lastLines = 0
$timeoutMin = 30

while ($true) {
    Start-Sleep -Seconds 15
    $elapsed = (Get-Date) - $start
    $elapsedMin = [math]::Round($elapsed.TotalMinutes, 1)

    if (Test-Path $sentinel) {
        Write-Host "[$elapsedMin min] sentinel 已写，流水线完成" -ForegroundColor Green
        break
    }

    if ($proc.HasExited) {
        Write-Host "[$elapsedMin min] admin 进程已退出 (exit $($proc.ExitCode))"
        break
    }

    if ($elapsed.TotalMinutes -gt $timeoutMin) {
        Write-Host "[$elapsedMin min] TIMEOUT $timeoutMin 分钟" -ForegroundColor Red
        break
    }

    if (Test-Path $logFile) {
        $lines = (Get-Content $logFile -ErrorAction SilentlyContinue).Count
        $newLines = $lines - $lastLines
        $lastLines = $lines
        Write-Host "[$elapsedMin min] 进度: log $lines 行 (+$newLines)" -ForegroundColor DarkGray
        if ($newLines -gt 0) {
            Get-Content $logFile -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object {
                Write-Host "  | $_" -ForegroundColor DarkGray
            }
        }
    } else {
        Write-Host "[$elapsedMin min] 等任务初始化..." -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "=== Sentinel ===" -ForegroundColor Cyan
if (Test-Path $sentinel) { Get-Content $sentinel } else { Write-Host "(无)" -ForegroundColor Red }

Write-Host ""
Write-Host "=== 日志末尾 ===" -ForegroundColor Cyan
if (Test-Path $logFile) { Get-Content $logFile -Tail 80 }

Write-Host ""
Write-Host "=== results/ 下 ncu_* 产物 ===" -ForegroundColor Cyan
Get-ChildItem 'D:\Desktop\black hole\results' -Filter 'ncu_*' -ErrorAction SilentlyContinue |
    Sort-Object Name |
    Select-Object Name, @{N='SizeKB';E={[math]::Round($_.Length/1KB, 1)}}, LastWriteTime |
    Format-Table -AutoSize
