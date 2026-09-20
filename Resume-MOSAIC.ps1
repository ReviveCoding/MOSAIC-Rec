$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$Repo = "C:\Users\bjw-0\Downloads\MOSAIC-Rec"
$Scratch = "\\wsl.localhost\Ubuntu-22.04\home\bjw-0\mosaic-rec-work"

Set-Location $Repo

$CodexArgs = @(
    "--approve-for-me",
    "--cd", $Repo,
    "--model", "gpt-5.6-sol",
    "-c", 'model_reasoning_effort="medium"',
    "-c", 'model_verbosity="low"',
    "-c", 'web_search="live"',
    "-c", 'sandbox_workspace_write.network_access=true'
)

if (
    $Scratch -and
    $Scratch -ne "__NONE__" -and
    (Test-Path -LiteralPath $Scratch)
) {
    $CodexArgs += @("--add-dir", $Scratch)
}

& codex @CodexArgs resume --last
