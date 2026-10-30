$ErrorActionPreference = 'Stop'
$env:KMP_DUPLICATE_LIB_OK = 'TRUE'
$env:PYTHONUNBUFFERED = '1'

$Root = 'G:\CT_RE\CT_DRA'
$Python = 'G:\CT_RE\conda\envs\ct_pnp\python.exe'
$StatusPath = Join-Path $Root 'outputs\ablation\c_multiseed_pipeline_status.json'
$LogPath = Join-Path $Root 'outputs\ablation\c_multiseed_pipeline.log'
$ErrorLogPath = Join-Path $Root 'outputs\ablation\c_multiseed_pipeline_error.log'

function Write-PipelineStatus {
    param(
        [string]$State,
        [string]$Stage,
        [string]$Message = ''
    )
    $payload = [ordered]@{
        state = $State
        stage = $Stage
        message = $Message
        updated_at = (Get-Date).ToString('o')
        c_seed_20260917_status = (Join-Path $Root 'outputs\ablation\csnet_compressed_no_hr_skip\seed_20260917\status.json')
        c_seed_20260918_status = (Join-Path $Root 'outputs\ablation\csnet_compressed_no_hr_skip\seed_20260918\status.json')
        comparison_dir = (Join-Path $Root 'outputs\ablation\summary\c_vs_d_multiseed')
        cleanup_inventory = (Join-Path $Root 'outputs\cleanup_inventory_20261020')
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-PipelineStatus -State 'running' -Stage $Name
    "[$((Get-Date).ToString('o'))] START $Name" | Add-Content -LiteralPath $LogPath -Encoding UTF8
    & $Python @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    "[$((Get-Date).ToString('o'))] END $Name" | Add-Content -LiteralPath $LogPath -Encoding UTF8
}

try {
    Write-PipelineStatus -State 'running' -Stage 'training_C_seeds_20260917_20260918'
    Invoke-Step -Name 'training_C_seeds_20260917_20260918' -Arguments @(
        (Join-Path $Root 'run_ablation.py'),
        '--experiments', 'csnet_compressed_no_hr_skip',
        '--seeds', '20260917', '20260918'
    )

    Invoke-Step -Name 'evaluate_C_seeds_on_PCB7_PCB8' -Arguments @(
        (Join-Path $Root 'evaluate_pcb7_pcb8.py'),
        '--boards', '7', '8',
        '--skip-fdk',
        '--methods',
        'C_csnet_compressed_no_hr_skip_seed_20260917',
        'C_csnet_compressed_no_hr_skip_seed_20260918'
    )

    Invoke-Step -Name 'refresh_cross_board_summary' -Arguments @(
        (Join-Path $Root 'analyze_pcb7_pcb8_results.py')
    )

    Invoke-Step -Name 'compare_C_and_D_three_seeds' -Arguments @(
        (Join-Path $Root 'compare_c_d_multiseed.py')
    )

    Invoke-Step -Name 'inventory_cleanup_candidates_without_deleting' -Arguments @(
        (Join-Path $Root 'inventory_experiment_cleanup.py')
    )

    Write-PipelineStatus -State 'completed' -Stage 'all_done' -Message 'C three-seed training, PCB6/7/8 evaluation, C-vs-D comparison, and cleanup inventory completed.'
}
catch {
    $details = $_ | Out-String
    $details | Add-Content -LiteralPath $ErrorLogPath -Encoding UTF8
    Write-PipelineStatus -State 'failed' -Stage 'pipeline_error' -Message $_.Exception.Message
    throw
}
