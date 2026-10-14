#requires -Version 5.1
<#
Lab 7 - Automated Semantic Cache Threshold Experiment

Purpose:
  Evaluate semantic-cache behavior across thresholds without deleting the
  persistent exact cache.

Thresholds:
  0.70, 0.75, 0.80, 0.85, 0.90, 0.95

Per threshold:
  - 12 seed -> positive paraphrase pairs
  - 12 hard-negative probes
  - 6 unrelated probes
  = 42 API requests / threshold
  = 252 API requests total

The script:
  1. Requires port 8000 to be free before starting.
  2. Starts uvicorn with SEMANTIC_THRESHOLD set for that run.
  3. Verifies /health.
  4. Sends requests and classifies each request as EXACT / SEMANTIC / MISS
     using /metrics counter deltas.
  5. Saves raw results and summary CSV/JSON.
  6. Stops the server after each threshold.
  7. Restores the parent PowerShell environment to the original threshold value.

IMPORTANT:
  - Do NOT delete .aip_cache or calls.sqlite3.
  - Stop any manually running uvicorn on port 8000 before launching this script.
  - A semantic hit on a hard-negative case is flagged POTENTIAL_FALSE_SEMANTIC_HIT.
    The script does not claim it is definitely wrong; inspect the saved answer.
#>

$ErrorActionPreference = "Stop"

# -----------------------------
# Configuration
# -----------------------------
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

$HostAddress = "127.0.0.1"
$Port = 8000
$BaseUrl = "http://$HostAddress`:$Port"

$Thresholds = @(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

$ReportsDir = Join-Path $ProjectRoot "reports"
$StdoutDir = Join-Path $ReportsDir "semantic_cache_experiment_logs"

$RunId = Get-Date -Format "yyyyMMdd-HHmmss"

$RawCsv = Join-Path $ReportsDir "lab7_semantic_cache_results.csv"
$SummaryCsv = Join-Path $ReportsDir "lab7_semantic_cache_summary.csv"
$JsonPath = Join-Path $ReportsDir "lab7_semantic_cache_results.json"

New-Item -ItemType Directory -Force -Path $ReportsDir | Out-Null
New-Item -ItemType Directory -Force -Path $StdoutDir | Out-Null

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

# -----------------------------
# Test corpus
# -----------------------------
# Each pair has:
#   Seed      = deliberately stored in semantic cache
#   Positive  = same intent, paraphrased
#   Negative  = similar vocabulary but different intent
#
# The pair marker is shared only within the pair. This reduces accidental
# cross-pair matching while keeping seed/probe wording semantically related.
$Pairs = @(
    @{
        Id = "P01"
        Seed = "What is the deadline for submitting a reimbursement claim?"
        Positive = "After discharge, how long do I have to file my reimbursement?"
        Negative = "What documents are required to submit a reimbursement claim?"
    },
    @{
        Id = "P02"
        Seed = "How many days do I have to respond to an Aurora query?"
        Positive = "What is the allowed time to answer an Aurora query?"
        Negative = "What documents are needed when Aurora raises a query?"
    },
    @{
        Id = "P03"
        Seed = "What is the renewal grace period?"
        Positive = "How long can I wait after the renewal date before renewing?"
        Negative = "How often must the policy be renewed?"
    },
    @{
        Id = "P04"
        Seed = "What is the grievance escalation procedure?"
        Positive = "How do I escalate a complaint that has not been resolved?"
        Negative = "Who handles an escalated grievance?"
    },
    @{
        Id = "P05"
        Seed = "When is KYC documentation required?"
        Positive = "Under what circumstances do I need to submit KYC documents?"
        Negative = "What information is included in KYC documentation?"
    },
    @{
        Id = "P06"
        Seed = "How do I submit a reimbursement claim?"
        Positive = "What is the process for filing a reimbursement claim?"
        Negative = "What is the deadline for filing the reimbursement claim?"
    },
    @{
        Id = "P07"
        Seed = "What happens after a claim is queried?"
        Positive = "What happens when Aurora raises a query on my claim?"
        Negative = "How long do I have to respond after a claim query?"
    },
    @{
        Id = "P08"
        Seed = "How do I upload claim documents?"
        Positive = "What is the procedure for uploading documents for my claim?"
        Negative = "Which documents are required for the claim?"
    },
    @{
        Id = "P09"
        Seed = "What happens if a claim misses the deadline?"
        Positive = "What occurs when I submit my reimbursement claim late?"
        Negative = "What is the normal deadline for submitting the claim?"
    },
    @{
        Id = "P10"
        Seed = "How can I reopen a closed claim?"
        Positive = "What is the process for reopening a claim that was closed?"
        Negative = "Why would Aurora close a claim?"
    },
    @{
        Id = "P11"
        Seed = "What is the process for filing a grievance?"
        Positive = "How can I submit a complaint through the grievance process?"
        Negative = "How do I escalate a grievance that has already been filed?"
    },
    @{
        Id = "P12"
        Seed = "What is the post-hospitalisation submission period?"
        Positive = "How long after the post-hospitalisation period can I submit those bills?"
        Negative = "What is the deadline for the main reimbursement claim?"
    }
)

$Unrelated = @(
    "What is gradient descent?",
    "How does PCA reduce dimensionality?",
    "What is the difference between classification and regression?",
    "How does a decision tree select a split?",
    "What is an eigenvalue?",
    "What is a convolutional neural network?"
)

# -----------------------------
# Helpers
# -----------------------------
function Get-Health {
    return Invoke-RestMethod -Method Get -Uri "$BaseUrl/health" -TimeoutSec 30
}

function Get-Metrics {
    return Invoke-RestMethod -Method Get -Uri "$BaseUrl/metrics" -TimeoutSec 30
}

function Wait-ForHealth {
    param(
        [double]$ExpectedThreshold,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)

    while ((Get-Date) -lt $deadline) {
        try {
            $h = Get-Health

            if ($h.status -eq "ok") {
                $actual = [double]$h.semantic_threshold
                $entries = [int]$h.semantic_cache_entries

                if ([Math]::Abs($actual - $ExpectedThreshold) -lt 0.00001 -and $entries -eq 0) {
                    return $h
                }

                Write-Host ("Waiting for clean semantic cache: threshold={0}, entries={1}" -f $actual, $entries)
            }
        }
        catch {
            # Server may still be starting.
        }

        Start-Sleep -Milliseconds 500
    }

    throw "Server did not become healthy with threshold $ExpectedThreshold and semantic_cache_entries=0 within $TimeoutSeconds seconds."
}

function Invoke-Ask {
    param(
        [string]$Question
    )

    $body = @{
        question = $Question
        top_k = 5
        mode = "rag"
    } | ConvertTo-Json -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "$BaseUrl/ask" `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec 120

        $sw.Stop()

        return @{
            Success = $true
            Response = $response
            ClientLatencyMs = [Math]::Round($sw.Elapsed.TotalMilliseconds, 3)
            Error = $null
        }
    }
    catch {
        $sw.Stop()

        return @{
            Success = $false
            Response = $null
            ClientLatencyMs = [Math]::Round($sw.Elapsed.TotalMilliseconds, 3)
            Error = $_.Exception.Message
        }
    }
}

function Get-CacheCounters {
    param(
        $Metrics
    )

    return @{
        Exact = [int]$Metrics.cache.exact_hits
        Semantic = [int]$Metrics.cache.semantic_hits
        Misses = [int]$Metrics.cache.misses
    }
}

function Get-CacheType {
    param(
        $Before,
        $After
    )

    $dExact = [int]$After.Exact - [int]$Before.Exact
    $dSemantic = [int]$After.Semantic - [int]$Before.Semantic
    $dMiss = [int]$After.Misses - [int]$Before.Misses

    if ($dExact -eq 1 -and $dSemantic -eq 0 -and $dMiss -eq 0) {
        return "EXACT"
    }

    if ($dExact -eq 0 -and $dSemantic -eq 1 -and $dMiss -eq 0) {
        return "SEMANTIC"
    }

    if ($dExact -eq 0 -and $dSemantic -eq 0 -and $dMiss -eq 1) {
        return "MISS"
    }

    return "UNKNOWN"
}

function Get-AnswerText {
    param($Response)

    if ($null -eq $Response) {
        return ""
    }

    $answer = [string]$Response.answer

    if ($answer.Length -gt 1000) {
        return $answer.Substring(0, 1000) + "..."
    }

    return $answer
}

function Add-Result {
    param(
        [System.Collections.Generic.List[object]]$Results,
        [string]$Threshold,
        [string]$CaseId,
        [string]$Category,
        [string]$Role,
        [string]$Question,
        [string]$CacheType,
        [bool]$Cached,
        [bool]$Refused,
        [double]$LatencyMs,
        [double]$CostUsd,
        [string]$TraceId,
        [string]$Answer,
        [bool]$PotentialFalseSemanticHit,
        [string]$Error
    )

    $Results.Add([PSCustomObject]@{
        run_id = $RunId
        threshold = [double]$Threshold
        case_id = $CaseId
        category = $Category
        role = $Role
        question = $Question
        cache_type = $CacheType
        cached = $Cached
        refused = $Refused
        latency_ms = $LatencyMs
        cost_usd = $CostUsd
        trace_id = $TraceId
        potential_false_semantic_hit = $PotentialFalseSemanticHit
        answer = $Answer
        error = $Error
    })
}

# -----------------------------
# Safety check: port must be free
# -----------------------------
try {
    $existing = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue

    if ($existing) {
        $pids = ($existing | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
        throw "Port $Port is already in use by PID(s): $pids. Stop the manually running uvicorn first with Ctrl+C, then run this script again."
    }
}
catch {
    if ($_.Exception.Message -like "Port $Port is already in use*") {
        throw
    }
    # Get-NetTCPConnection may not exist on some environments; continue if it
    # failed for a command-availability reason.
}

# -----------------------------
# Main experiment
# -----------------------------
$Results = [System.Collections.Generic.List[object]]::new()
$Summary = [System.Collections.Generic.List[object]]::new()

$OriginalThreshold = $env:SEMANTIC_THRESHOLD
$ServerProcess = $null

try {
    foreach ($threshold in $Thresholds) {

        $thresholdText = $threshold.ToString("0.00")
        $tag = "SC-$thresholdText-$RunId".Replace(".", "")

        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host "SEMANTIC CACHE THRESHOLD: $thresholdText" -ForegroundColor Cyan
        Write-Host "Run tag: $tag"
        Write-Host "============================================================" -ForegroundColor Cyan

        # Set only the child server's environment through the current process.
        $env:SEMANTIC_THRESHOLD = $thresholdText

        $stdout = Join-Path $StdoutDir "server-$thresholdText-$RunId.out.log"
        $stderr = Join-Path $StdoutDir "server-$thresholdText-$RunId.err.log"

        $ServerProcess = Start-Process `
            -FilePath $PythonExe `
            -ArgumentList "-m uvicorn labs.lab7.service:app --host $HostAddress --port $Port" `
            -WorkingDirectory $ProjectRoot `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru

        Write-Host "Started server PID $($ServerProcess.Id). Waiting for health..."

        $health = Wait-ForHealth -ExpectedThreshold $threshold

        Write-Host ("Health OK: threshold={0}; semantic entries={1}" -f $health.semantic_threshold, $health.semantic_cache_entries) -ForegroundColor Green

        $thresholdStartMetrics = Get-Metrics
        $thresholdStart = Get-CacheCounters $thresholdStartMetrics

        # -------------------------
        # 12 seed + positive + negative groups
        # -------------------------
        foreach ($pair in $Pairs) {

            $pairTag = "$tag-$($pair.Id)"

            # Seed
            $seedQuestion = "[$pairTag] $($pair.Seed)"

            $beforeMetrics = Get-CacheCounters (Get-Metrics)
            $r = Invoke-Ask -Question $seedQuestion
            $afterMetrics = Get-CacheCounters (Get-Metrics)

            $cacheType = if ($r.Success) { Get-CacheType -Before $beforeMetrics -After $afterMetrics } else { "ERROR" }

            $answer = if ($r.Success) { Get-AnswerText $r.Response } else { "" }
            $cached = if ($r.Success) { [bool]$r.Response.cached } else { $false }
            $refused = if ($r.Success) { [bool]$r.Response.refused } else { $false }
            $cost = if ($r.Success) { [double]$r.Response.cost_usd } else { 0.0 }
            $trace = if ($r.Success) { [string]$r.Response.trace_id } else { "" }

            Add-Result `
                -Results $Results `
                -Threshold $threshold `
                -CaseId $pair.Id `
                -Category "POSITIVE_PAIR" `
                -Role "SEED" `
                -Question $seedQuestion `
                -CacheType $cacheType `
                -Cached $cached `
                -Refused $refused `
                -LatencyMs $r.ClientLatencyMs `
                -CostUsd $cost `
                -TraceId $trace `
                -Answer $answer `
                -PotentialFalseSemanticHit $false `
                -Error $r.Error

            Write-Host ("{0} SEED     {1,-8} {2,7} ms" -f $pair.Id, $cacheType, [int]$r.ClientLatencyMs)

            # Positive paraphrase
            $positiveQuestion = "[$pairTag] $($pair.Positive)"

            $beforeMetrics = Get-CacheCounters (Get-Metrics)
            $r = Invoke-Ask -Question $positiveQuestion
            $afterMetrics = Get-CacheCounters (Get-Metrics)

            $cacheType = if ($r.Success) { Get-CacheType -Before $beforeMetrics -After $afterMetrics } else { "ERROR" }

            $answer = if ($r.Success) { Get-AnswerText $r.Response } else { "" }
            $cached = if ($r.Success) { [bool]$r.Response.cached } else { $false }
            $refused = if ($r.Success) { [bool]$r.Response.refused } else { $false }
            $cost = if ($r.Success) { [double]$r.Response.cost_usd } else { 0.0 }
            $trace = if ($r.Success) { [string]$r.Response.trace_id } else { "" }

            Add-Result `
                -Results $Results `
                -Threshold $threshold `
                -CaseId $pair.Id `
                -Category "POSITIVE_PAIR" `
                -Role "POSITIVE_PROBE" `
                -Question $positiveQuestion `
                -CacheType $cacheType `
                -Cached $cached `
                -Refused $refused `
                -LatencyMs $r.ClientLatencyMs `
                -CostUsd $cost `
                -TraceId $trace `
                -Answer $answer `
                -PotentialFalseSemanticHit $false `
                -Error $r.Error

            Write-Host ("{0} POSITIVE  {1,-8} {2,7} ms" -f $pair.Id, $cacheType, [int]$r.ClientLatencyMs)

            # Hard negative
            $negativeQuestion = "[$pairTag] $($pair.Negative)"

            $beforeMetrics = Get-CacheCounters (Get-Metrics)
            $r = Invoke-Ask -Question $negativeQuestion
            $afterMetrics = Get-CacheCounters (Get-Metrics)

            $cacheType = if ($r.Success) { Get-CacheType -Before $beforeMetrics -After $afterMetrics } else { "ERROR" }

            $answer = if ($r.Success) { Get-AnswerText $r.Response } else { "" }
            $cached = if ($r.Success) { [bool]$r.Response.cached } else { $false }
            $refused = if ($r.Success) { [bool]$r.Response.refused } else { $false }
            $cost = if ($r.Success) { [double]$r.Response.cost_usd } else { 0.0 }
            $trace = if ($r.Success) { [string]$r.Response.trace_id } else { "" }

            $potentialFalse = ($cacheType -eq "SEMANTIC")

            Add-Result `
                -Results $Results `
                -Threshold $threshold `
                -CaseId $pair.Id `
                -Category "HARD_NEGATIVE" `
                -Role "NEGATIVE_PROBE" `
                -Question $negativeQuestion `
                -CacheType $cacheType `
                -Cached $cached `
                -Refused $refused `
                -LatencyMs $r.ClientLatencyMs `
                -CostUsd $cost `
                -TraceId $trace `
                -Answer $answer `
                -PotentialFalseSemanticHit $potentialFalse `
                -Error $r.Error

            if ($potentialFalse) {
                Write-Host ("{0} NEGATIVE  SEMANTIC  <-- POTENTIAL FALSE HIT" -f $pair.Id) -ForegroundColor Yellow
            }
            else {
                Write-Host ("{0} NEGATIVE  {1,-8} {2,7} ms" -f $pair.Id, $cacheType, [int]$r.ClientLatencyMs)
            }
        }

        # -------------------------
        # 6 unrelated probes
        # -------------------------
        $uIndex = 1

        foreach ($questionBase in $Unrelated) {
            $caseId = "U{0:D2}" -f $uIndex
            $question = "[$tag-$caseId] $questionBase"

            $beforeMetrics = Get-CacheCounters (Get-Metrics)
            $r = Invoke-Ask -Question $question
            $afterMetrics = Get-CacheCounters (Get-Metrics)

            $cacheType = if ($r.Success) { Get-CacheType -Before $beforeMetrics -After $afterMetrics } else { "ERROR" }

            $answer = if ($r.Success) { Get-AnswerText $r.Response } else { "" }
            $cached = if ($r.Success) { [bool]$r.Response.cached } else { $false }
            $refused = if ($r.Success) { [bool]$r.Response.refused } else { $false }
            $cost = if ($r.Success) { [double]$r.Response.cost_usd } else { 0.0 }
            $trace = if ($r.Success) { [string]$r.Response.trace_id } else { "" }

            $potentialFalse = ($cacheType -eq "SEMANTIC")

            Add-Result `
                -Results $Results `
                -Threshold $threshold `
                -CaseId $caseId `
                -Category "UNRELATED" `
                -Role "UNRELATED_PROBE" `
                -Question $question `
                -CacheType $cacheType `
                -Cached $cached `
                -Refused $refused `
                -LatencyMs $r.ClientLatencyMs `
                -CostUsd $cost `
                -TraceId $trace `
                -Answer $answer `
                -PotentialFalseSemanticHit $potentialFalse `
                -Error $r.Error

            if ($potentialFalse) {
                Write-Host ("{0} UNRELATED SEMANTIC <-- POTENTIAL FALSE HIT" -f $caseId) -ForegroundColor Yellow
            }
            else {
                Write-Host ("{0} UNRELATED  {1,-8} {2,7} ms" -f $caseId, $cacheType, [int]$r.ClientLatencyMs)
            }

            $uIndex++
        }

        # -------------------------
        # Threshold summary
        # -------------------------
        $thresholdRows = @($Results | Where-Object { [double]$_.threshold -eq [double]$threshold })
        $positiveProbes = @($thresholdRows | Where-Object { $_.role -eq "POSITIVE_PROBE" })
        $negativeProbes = @($thresholdRows | Where-Object { $_.role -eq "NEGATIVE_PROBE" })
        $unrelatedProbes = @($thresholdRows | Where-Object { $_.role -eq "UNRELATED_PROBE" })

        $positiveSemanticHits = @($positiveProbes | Where-Object { $_.cache_type -eq "SEMANTIC" }).Count
        $negativeSemanticHits = @($negativeProbes | Where-Object { $_.cache_type -eq "SEMANTIC" }).Count
        $unrelatedSemanticHits = @($unrelatedProbes | Where-Object { $_.cache_type -eq "SEMANTIC" }).Count

        $positiveHitRate = if ($positiveProbes.Count) {
            [Math]::Round($positiveSemanticHits / $positiveProbes.Count, 4)
        } else { 0 }

        $negativeFalseRate = if ($negativeProbes.Count) {
            [Math]::Round($negativeSemanticHits / $negativeProbes.Count, 4)
        } else { 0 }

        $unrelatedFalseRate = if ($unrelatedProbes.Count) {
            [Math]::Round($unrelatedSemanticHits / $unrelatedProbes.Count, 4)
        } else { 0 }

        $thresholdMetrics = Get-Metrics

        $Summary.Add([PSCustomObject]@{
            run_id = $RunId
            threshold = [double]$threshold
            total_requests = $thresholdRows.Count
            positive_probes = $positiveProbes.Count
            positive_semantic_hits = $positiveSemanticHits
            positive_semantic_hit_rate = $positiveHitRate
            hard_negative_probes = $negativeProbes.Count
            hard_negative_semantic_hits = $negativeSemanticHits
            hard_negative_false_hit_rate = $negativeFalseRate
            unrelated_probes = $unrelatedProbes.Count
            unrelated_semantic_hits = $unrelatedSemanticHits
            unrelated_false_hit_rate = $unrelatedFalseRate
            exact_hits = [int]$thresholdMetrics.cache.exact_hits - [int]$thresholdStart.Exact
            semantic_hits = [int]$thresholdMetrics.cache.semantic_hits - [int]$thresholdStart.Semantic
            misses = [int]$thresholdMetrics.cache.misses - [int]$thresholdStart.Misses
            total_cost_usd = [Math]::Round(
                (($thresholdRows | Measure-Object -Property cost_usd -Sum).Sum),
                6
            )
            avg_latency_ms = [Math]::Round(
                (($thresholdRows | Measure-Object -Property latency_ms -Average).Average),
                3
            )
            potential_false_semantic_hits = @($thresholdRows | Where-Object { $_.potential_false_semantic_hit -eq $true }).Count
        })

        Write-Host ""
        Write-Host "Threshold $thresholdText summary:" -ForegroundColor Cyan
        Write-Host ("  Positive semantic hit rate : {0:P1}" -f $positiveHitRate)
        Write-Host ("  Hard-negative semantic hits: {0}/{1}" -f $negativeSemanticHits, $negativeProbes.Count)
        Write-Host ("  Unrelated semantic hits    : {0}/{1}" -f $unrelatedSemanticHits, $unrelatedProbes.Count)
        Write-Host ("  Potential false hits       : {0}" -f @($thresholdRows | Where-Object { $_.potential_false_semantic_hit -eq $true }).Count)

        # Stop this threshold's server.
        if ($ServerProcess -and -not $ServerProcess.HasExited) {
            Stop-Process -Id $ServerProcess.Id -Force
            $ServerProcess.WaitForExit()
        }

        $ServerProcess = $null
        Start-Sleep -Seconds 1
    }
}
finally {
    if ($ServerProcess -and -not $ServerProcess.HasExited) {
        Stop-Process -Id $ServerProcess.Id -Force -ErrorAction SilentlyContinue
    }

    if ($null -eq $OriginalThreshold) {
        Remove-Item Env:SEMANTIC_THRESHOLD -ErrorAction SilentlyContinue
    }
    else {
        $env:SEMANTIC_THRESHOLD = $OriginalThreshold
    }
}

# -----------------------------
# Save outputs
# -----------------------------
$Results | Export-Csv -Path $RawCsv -NoTypeInformation -Encoding UTF8
$Summary | Export-Csv -Path $SummaryCsv -NoTypeInformation -Encoding UTF8

$JsonObject = [PSCustomObject]@{
    run_id = $RunId
    thresholds = $Thresholds
    request_count = $Results.Count
    scenario_count = 24
    results = $Results
    summary = $Summary
}

$JsonObject | ConvertTo-Json -Depth 8 | Set-Content -Path $JsonPath -Encoding UTF8

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "EXPERIMENT COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Requests completed : $($Results.Count)"
Write-Host "Raw results        : $RawCsv"
Write-Host "Summary            : $SummaryCsv"
Write-Host "JSON               : $JsonPath"
Write-Host ""
Write-Host "Threshold summary:"
$Summary | Format-Table threshold, positive_semantic_hit_rate, hard_negative_semantic_hits, unrelated_semantic_hits, potential_false_semantic_hits, total_cost_usd, avg_latency_ms -AutoSize
Write-Host ""
Write-Host "Review rows with potential_false_semantic_hit = True before selecting a threshold."
