#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Enable GPU profiling for Nsight Compute / ncu in WSL2.
    Fixes ERR_NVGPUCTRPERM.

.NOTES
    Run this script as Administrator. After running, restart WSL2:
        wsl --shutdown
#>

$regPath = "HKLM:\SOFTWARE\NVIDIA Corporation\GPU Profiling"
if (-not (Test-Path $regPath)) {
    New-Item -Path $regPath -Force | Out-Null
}
Set-ItemProperty -Path $regPath -Name "EnableGpuProfiling" -Value 1 -Type DWord
Write-Host "GPU profiling enabled. Restart WSL2 with: wsl --shutdown" -ForegroundColor Green
