param(
    [string]$InputRoot = "Round 1 result",
    [string]$OutputRoot = "prepared_round_1_import"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.IO.Compression.FileSystem

function Ensure-Dir {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
}

function Get-XlsxRows {
    param([string]$Path)

    $resolved = Resolve-Path -LiteralPath $Path
    $zip = [System.IO.Compression.ZipFile]::OpenRead($resolved)
    try {
        $shared = @()
        $sharedEntry = $zip.Entries | Where-Object { $_.FullName -eq "xl/sharedStrings.xml" }
        if ($sharedEntry) {
            $reader = New-Object IO.StreamReader($sharedEntry.Open())
            try {
                $xml = [xml]$reader.ReadToEnd()
            }
            finally {
                $reader.Close()
            }
            foreach ($si in $xml.sst.si) {
                if ($si.t) {
                    $shared += [string]$si.t
                }
                elseif ($si.r) {
                    $shared += (($si.r | ForEach-Object { $_.t.'#text' }) -join "")
                }
                else {
                    $shared += ""
                }
            }
        }

        $sheetEntry = $zip.Entries | Where-Object { $_.FullName -eq "xl/worksheets/sheet1.xml" }
        if (-not $sheetEntry) {
            throw "Workbook '$Path' does not contain xl/worksheets/sheet1.xml"
        }

        $reader = New-Object IO.StreamReader($sheetEntry.Open())
        try {
            $sheet = [xml]$reader.ReadToEnd()
        }
        finally {
            $reader.Close()
        }

        $rows = @($sheet.worksheet.sheetData.row)
        if ($rows.Count -eq 0) {
            return @()
        }

        $header = @()
        foreach ($cell in $rows[0].c) {
            $value = $cell.v
            if ($cell.t -eq "s" -and $null -ne $value) {
                $value = $shared[[int]$value]
            }
            $header += [string]$value
        }

        $output = @()
        foreach ($row in ($rows | Select-Object -Skip 1)) {
            $values = @()
            foreach ($cell in $row.c) {
                $value = $cell.v
                if ($cell.t -eq "s" -and $null -ne $value) {
                    $value = $shared[[int]$value]
                }
                $values += $value
            }
            if (($values | Where-Object { $_ -ne $null -and $_ -ne "" }).Count -eq 0) {
                continue
            }
            $obj = [ordered]@{}
            for ($i = 0; $i -lt $header.Count; $i++) {
                $key = $header[$i]
                if (-not $key) {
                    $key = "column_$i"
                }
                $obj[$key] = if ($i -lt $values.Count) { $values[$i] } else { $null }
            }
            $output += [pscustomobject]$obj
        }
        return $output
    }
    finally {
        $zip.Dispose()
    }
}

function Get-FieldValue {
    param(
        [pscustomobject]$Row,
        [string[]]$CandidateNames
    )

    foreach ($property in $Row.PSObject.Properties) {
        foreach ($candidate in $CandidateNames) {
            if ($property.Name -ieq $candidate) {
                return [string]$property.Value
            }
        }
    }
    return $null
}

function Parse-TimeToSeconds {
    param([string]$Raw)

    $text = ([string]$Raw).Trim().ToLowerInvariant()
    if (-not $text) {
        throw "Empty time string"
    }

    if ($text -notmatch '^(?:(?<min>\d+)\s*min)?\s*(?:(?<sec>\d+)\s*s)?$') {
        throw "Unsupported time string '$Raw'"
    }

    $minutes = if ($Matches["min"]) { [int]$Matches["min"] } else { 0 }
    $seconds = if ($Matches["sec"]) { [int]$Matches["sec"] } else { 0 }
    return ($minutes * 60) + $seconds
}

function Get-CaseIdFromFileName {
    param([string]$Name)
    return (($Name -replace '\.nii\.gz$', '') -replace '_revised(_phase1|_final)?$', '')
}

function Get-KeyFromCaseId {
    param([string]$CaseId)

    if ($CaseId -match 'Rat(?<rat>\d+)_M_1_(?<minor>\d+)_') {
        return ('{0},1-{1}' -f $Matches["rat"], $Matches["minor"])
    }
    if ($CaseId -match '_MHR_(?<rat>\d+)_M_I_(?<tp>D\d+|M\d+)_') {
        return ('{0}.{1}' -f $Matches["rat"], $Matches["tp"].ToUpperInvariant())
    }
    throw "Unable to derive lookup key from case id '$CaseId'"
}

function Normalize-RatId {
    param([string]$RatId)

    $raw = ([string]$RatId).Trim().ToUpperInvariant()
    $compact = $raw -replace '\s+', '' -replace ',', '.'

    if ($compact -match '^(?<rat>\d+)\.1-(?<minor>\d+)$') {
        return ('{0},1-{1}' -f $Matches["rat"], $Matches["minor"])
    }
    if ($compact -match '^(?<rat>\d+)\.M\d+_(?<minor>\d+)$') {
        return ('{0},1-{1}' -f $Matches["rat"], $Matches["minor"])
    }
    if ($compact -match '^(?<rat>\d+)(?<tp>D\d+|M\d+)$') {
        return ('{0}.{1}' -f $Matches["rat"], $Matches["tp"])
    }
    if ($compact -match '^(?<rat>\d+)\.(?<tp>D\d+|M\d+)$') {
        return ('{0}.{1}' -f $Matches["rat"], $Matches["tp"])
    }

    throw "Unable to normalize RAT_ID '$RatId'"
}

function Build-LabelMap {
    param([string]$LabelDir)

    $map = @{}
    foreach ($file in (Get-ChildItem -LiteralPath $LabelDir -File -Filter *.nii.gz | Sort-Object Name)) {
        $caseId = Get-CaseIdFromFileName $file.Name
        $key = Get-KeyFromCaseId $caseId
        if ($map.ContainsKey($key)) {
            throw "Duplicate label lookup key '$key' in '$LabelDir'"
        }
        $map[$key] = [pscustomobject]@{
            key = $key
            case_id = $caseId
            source_file = $file.FullName
            source_name = $file.Name
        }
    }
    return $map
}

function Write-Csv {
    param(
        [string]$Path,
        [object[]]$Rows
    )

    if ($Rows.Count -eq 0) {
        Set-Content -LiteralPath $Path -Value ""
        return
    }
    $Rows | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

$inputRootPath = Resolve-Path -LiteralPath $InputRoot
$outputRootPath = Join-Path (Get-Location) $OutputRoot
if (Test-Path -LiteralPath $outputRootPath) {
    throw "Output root '$outputRootPath' already exists. Remove it or choose a different -OutputRoot."
}

Ensure-Dir $outputRootPath

$manualKeyOverrides = @{
    "audit_anchor_input" = @{
        "2465.D28" = @{
            replacement = "2464.D28"
            reason = "Audit phase1 time sheet says 2465 D28, but both audit label folders use case 2464 D28 and audit phase2 time sheet also uses 2464 D28."
        }
    }
}

$stageSpecs = @(
    [pscustomobject]@{
        name = "routine_input"
        label_dir = Join-Path $inputRootPath "Routine final revised label"
        time_file = Join-Path $inputRootPath "Routine review time.xlsx"
        time_field = "review_time"
    },
    [pscustomobject]@{
        name = "audit_anchor_input"
        label_dir = Join-Path $inputRootPath "Audit phase1 revised label"
        time_file = Join-Path $inputRootPath "Audit phase1 time.xlsx"
        time_field = "anchor_time"
    },
    [pscustomobject]@{
        name = "audit_final_input"
        label_dir = Join-Path $inputRootPath "Audit final revised label"
        time_file = Join-Path $inputRootPath "Audit phase2 time.xlsx"
        time_field = "assisted_time"
    }
)

$summaryStages = @()

foreach ($stage in $stageSpecs) {
    $stageRoot = Join-Path $outputRootPath $stage.name
    $labelsOut = Join-Path $stageRoot "labels"
    Ensure-Dir $stageRoot
    Ensure-Dir $labelsOut

    $labelMap = Build-LabelMap $stage.label_dir
    $timeRows = Get-XlsxRows $stage.time_file
    $metadataRows = @()
    $stageWarnings = @()
    $seenCaseIds = @{}

    foreach ($row in $timeRows) {
        $ratId = Get-FieldValue $row @("RAT_ID", "rat_id", "Rat_ID")
        $timeText = Get-FieldValue $row @("TIME", "time", "TImE")
        if (-not $ratId) {
            throw "Missing RAT_ID in '$($stage.time_file)'"
        }
        if (-not $timeText) {
            throw "Missing time value for RAT_ID '$ratId' in '$($stage.time_file)'"
        }

        $normalizedKey = Normalize-RatId $ratId
        if ($manualKeyOverrides.ContainsKey($stage.name) -and $manualKeyOverrides[$stage.name].ContainsKey($normalizedKey)) {
            $override = $manualKeyOverrides[$stage.name][$normalizedKey]
            $stageWarnings += [pscustomobject]@{
                stage = $stage.name
                type = "manual_key_override"
                original_key = $normalizedKey
                replacement_key = $override.replacement
                rat_id = $ratId
                detail = $override.reason
            }
            $normalizedKey = $override.replacement
        }

        if (-not $labelMap.ContainsKey($normalizedKey)) {
            throw "No label file in '$($stage.label_dir)' matches RAT_ID '$ratId' (normalized '$normalizedKey')"
        }

        $label = $labelMap[$normalizedKey]
        if ($seenCaseIds.ContainsKey($label.case_id)) {
            throw "Duplicate metadata mapping for case '$($label.case_id)' in stage '$($stage.name)'"
        }

        $seconds = Parse-TimeToSeconds $timeText
        $destPath = Join-Path $labelsOut ($label.case_id + ".nii.gz")
        Copy-Item -LiteralPath $label.source_file -Destination $destPath -Force
        $seenCaseIds[$label.case_id] = $true

        $metadataRows += [pscustomobject]([ordered]@{
            case_id = $label.case_id
            $stage.time_field = $seconds
            source_ratid = $ratId
            normalized_key = $normalizedKey
            source_time_text = $timeText
            source_label_filename = $label.source_name
        })
    }

    $missingLabels = @($labelMap.Values | Where-Object { -not $seenCaseIds.ContainsKey($_.case_id) })
    if ($missingLabels.Count -gt 0) {
        throw "Stage '$($stage.name)' is missing time rows for labels: $($missingLabels.case_id -join ', ')"
    }

    $metadataRows = @($metadataRows | Sort-Object case_id)
    Write-Csv -Path (Join-Path $stageRoot "metadata.csv") -Rows $metadataRows

    $summaryStages += [pscustomobject]@{
        stage = $stage.name
        label_count = @($labelMap.Keys).Count
        metadata_count = $metadataRows.Count
        output_root = $stageRoot
        warnings = $stageWarnings
    }
}

$summary = [pscustomobject]@{
    source_root = [string]$inputRootPath
    output_root = [string]$outputRootPath
    created_at = (Get-Date).ToString("s")
    notes = @(
        "Labels were copied without changing voxel values.",
        "Filenames were normalized back to <case_id>.nii.gz by removing *_revised, *_revised_phase1, and *_revised_final suffixes.",
        "Time values were converted to integer seconds because the pipeline expects numeric metadata fields.",
        "Metadata CSV files include extra provenance columns; the pipeline will ignore them."
    )
    stages = $summaryStages
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $outputRootPath "summary.json") -Encoding UTF8

$readme = @"
Prepared round 1 import bundle

This folder was generated from:
  $inputRootPath

Pipeline-ready directories:
  - routine_input
  - audit_anchor_input
  - audit_final_input

Each directory contains:
  - labels/<case_id>.nii.gz
  - metadata.csv

Time units:
  - All metadata time values are stored as integer seconds.

Important notes:
  - Label voxels were not modified.
  - File names were normalized back to exact case ids.
  - One audit phase1 time row was corrected by explicit override:
    2465 D28 -> 2464.D28
    Reason: phase1 and phase2 audit label filenames both correspond to case 2464 D28.
"@
$readme | Set-Content -LiteralPath (Join-Path $outputRootPath "README.txt") -Encoding UTF8

Write-Host "Prepared pipeline-ready import bundle at: $outputRootPath"
