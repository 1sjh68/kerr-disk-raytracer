# 完整 ncu profiling 流水线（在 Windows admin 下跑）
# 通过 schtasks 触发，运行结束写 sentinel 文件供外部 poll
param(
    [string]$LogFile = "$env:TEMP\ncu_full_pipeline.log",
    [string]$Sentinel = "$env:TEMP\ncu_pipeline_done.txt"
)

# 写一个起始标记（防止外面 poll 误判）
"START $(Get-Date -Format o)" | Out-File "$Sentinel.start" -Encoding ascii

try {
    Start-Transcript -Path $LogFile -Force | Out-Null
    $ErrorActionPreference = 'Continue'

    $NcuExe  = 'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\target\windows-desktop-win7-x64\ncu.exe'
    $Py      = 'D:\Desktop\black hole\.venv\Scripts\python.exe'
    $Proj    = 'D:\Desktop\black hole'
    $Results = "$Proj\results"
    Set-Location $Proj

    Write-Host '=================================================='
    Write-Host ' ncu Full Profiling Pipeline'
    Write-Host '=================================================='
    Write-Host "ncu:    $NcuExe"
    Write-Host "python: $Py"
    Write-Host "proj:   $Proj"
    Write-Host "时间:   $(Get-Date)"
    Write-Host ''

    $admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] 'Administrator')
    Write-Host "admin: $admin"
    Write-Host "user:  $env:USERNAME"
    & $NcuExe --version | Select-Object -First 3
    Write-Host ''

    function Profile-Run {
        param(
            [string]$Tag,
            [string]$KernelName,
            [string[]]$PyArgs
        )
        $repBase = "$Results\ncu_$Tag"
        $rep     = "$repBase.ncu-rep"
        $sum     = "$repBase.summary.txt"
        $csv     = "$repBase.csv"
        $log     = "$repBase.run.log"

        Write-Host "----------"
        Write-Host "[$Tag] kernel=$KernelName  args=$($PyArgs -join ' ')"
        Write-Host "----------"
        $t0 = Get-Date

        & $NcuExe `
            --set full `
            --kernel-name $KernelName `
            --launch-skip 0 --launch-count 1 `
            --target-processes all `
            --export $repBase `
            --force-overwrite `
            $Py @PyArgs 2>&1 | Tee-Object -FilePath $log

        $dt = (Get-Date) - $t0
        Write-Host "[$Tag] 用时 $dt"

        if (Test-Path $rep) {
            Write-Host "[$Tag] 写 summary 到 $sum"
            & $NcuExe --import $rep --print-summary per-kernel 2>&1 | Out-File -FilePath $sum -Encoding utf8
            Write-Host "[$Tag] 写 csv 到 $csv"
            & $NcuExe --import $rep --csv --page details 2>&1 | Out-File -FilePath $csv -Encoding utf8
            $size = (Get-Item $rep).Length / 1MB
            Write-Host ("[$Tag] OK  rep={0:N2} MB" -f $size)
        } else {
            Write-Host "[$Tag] FAIL  rep 未生成"
        }
        Write-Host ''
    }

    Profile-Run -Tag 'float64_48' -KernelName 'kerr_geodesic_kernel_double' `
        -PyArgs @('run_geodesic_gpu.py', '--precision', 'float64')

    Profile-Run -Tag 'float64_128' -KernelName 'kerr_geodesic_kernel_double' `
        -PyArgs @('run_geodesic_gpu.py', '--precision', 'float64', '--resolution', '128')

    Profile-Run -Tag 'float32_48' -KernelName 'kerr_geodesic_kernel' `
        -PyArgs @('run_geodesic_gpu.py', '--precision', 'float32')

    Profile-Run -Tag 'float32_128' -KernelName 'kerr_geodesic_kernel' `
        -PyArgs @('run_geodesic_gpu.py', '--precision', 'float32', '--resolution', '128')

    Write-Host ''
    Write-Host '=================================================='
    Write-Host " 所有产物 in $Results"
    Write-Host '=================================================='
    Get-ChildItem $Results -Filter 'ncu_*' | Sort-Object Name | Select-Object Name, @{N='SizeKB';E={[math]::Round($_.Length/1KB,1)}}, LastWriteTime | Format-Table -AutoSize

    Stop-Transcript | Out-Null
    "DONE $(Get-Date -Format o)" | Out-File $Sentinel -Encoding ascii
} catch {
    "ERROR $(Get-Date -Format o) $_" | Out-File $Sentinel -Encoding ascii
    try { Stop-Transcript | Out-Null } catch {}
}
