# Sets ANTHROPIC_API_KEY as a permanent user environment variable on Windows.
# Run this ONCE in PowerShell after getting your key from console.anthropic.com
#
# Usage:   .\set_api_key.ps1 sk-ant-api03-xxxxxxxx
# After running, close and reopen PowerShell, then verify with:
#   echo $env:ANTHROPIC_API_KEY

param(
    [Parameter(Mandatory=$true)]
    [string]$ApiKey
)

if (-not $ApiKey.StartsWith("sk-ant-")) {
    Write-Host "Warning: Anthropic API keys usually start with 'sk-ant-'. Continuing anyway..." -ForegroundColor Yellow
}

[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", $ApiKey, "User")
$env:ANTHROPIC_API_KEY = $ApiKey

Write-Host "ANTHROPIC_API_KEY saved to your Windows user environment." -ForegroundColor Green
Write-Host "You can now run:" -ForegroundColor Cyan
Write-Host "  python magazine_grouper.py images" -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: any NEW PowerShell window will pick this up automatically." -ForegroundColor Gray
