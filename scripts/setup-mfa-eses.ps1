param(
    [string]$CondaCommand = "conda",
    [string]$EnvironmentName = "tts-mfa"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot ".env"

if (-not (Get-Command $CondaCommand -ErrorAction SilentlyContinue)) {
    throw "No se encontro conda. Instala Miniforge y vuelve a ejecutar este script."
}

& $CondaCommand create --yes --name $EnvironmentName --channel conda-forge montreal-forced-aligner
& $CondaCommand run --name $EnvironmentName mfa model download acoustic spanish_mfa
& $CondaCommand run --name $EnvironmentName mfa model download dictionary spanish_mfa

$environments = (& $CondaCommand env list --json | ConvertFrom-Json).envs
$environmentPath = $environments | Where-Object { $_ -like "*\\$EnvironmentName" } | Select-Object -First 1
if (-not $environmentPath) {
    throw "No se pudo localizar el entorno $EnvironmentName tras la instalacion."
}
$mfaExecutable = Join-Path $environmentPath "Scripts\\mfa.exe"
if (-not (Test-Path $mfaExecutable)) {
    throw "No se encontro mfa.exe en $mfaExecutable."
}

function Set-DotEnvValue([string]$Key, [string]$Value) {
    $line = "$Key=$Value"
    $contents = if (Test-Path $envFile) { Get-Content $envFile } else { @() }
    if ($contents -match "^$([regex]::Escape($Key))=") {
        $contents = $contents -replace "^$([regex]::Escape($Key))=.*$", $line
    } else {
        $contents += $line
    }
    Set-Content -Path $envFile -Value $contents -Encoding utf8
}

Set-DotEnvValue "TTS_EVAL_MFA_ENABLED" "true"
Set-DotEnvValue "TTS_EVAL_MFA_COMMAND" $mfaExecutable
Set-DotEnvValue "TTS_EVAL_MFA_ESES_ACOUSTIC_MODEL" "spanish_mfa"
Set-DotEnvValue "TTS_EVAL_MFA_ESES_DICTIONARY" "spanish_mfa"

Write-Host "MFA para es-ES activado. Reinicia Uvicorn para recargar .env."
