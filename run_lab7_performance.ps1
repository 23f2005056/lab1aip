# ============================================================
# LAB 7 - 50 UNCACHED PERFORMANCE EXPERIMENT
# ============================================================

$questions = @(
    "What is the current deadline for submitting a main reimbursement claim?",
    "From which date is the main reimbursement claim deadline calculated?",
    "How much time is allowed for post-hospitalisation bills?",
    "When must pre-hospitalisation expenses be submitted?",
    "How long does a claimant have to respond to an Aurora query?",
    "What happens if a claim submission is delayed beyond the stated deadline?",
    "Who decides whether a delayed claim can be condoned?",
    "How long does the claims committee have to decide a delay-condonation request?",

    "What is the procedure for submitting a reimbursement claim?",
    "What information needs to be provided when filing a claim?",
    "What should a claimant do after receiving a request for additional documents?",
    "What happens after Aurora asks a claimant for clarification?",
    "What should be checked before submitting a reimbursement claim?",
    "What information should accompany a claim submission?",
    "How does Aurora handle incomplete claim submissions?",
    "What should a claimant do if additional evidence is requested after filing?",

    "What documentation is associated with hospitalisation expenses?",
    "What records should be retained after a hospital stay for reimbursement purposes?",
    "How are hospitalisation-related expenses submitted with a claim?",
    "What supporting evidence may be required for hospital treatment?",
    "What should a claimant provide for expenses incurred before hospitalisation?",
    "What documentation may be needed for expenses incurred after hospitalisation?",
    "How are post-hospitalisation expenses handled under the claim process?",
    "What is the relationship between hospital discharge and the claim submission deadline?",

    "Which documents may be required to support a reimbursement claim?",
    "When would Aurora ask for additional claim documents?",
    "What should a claimant do if one of the required documents is unavailable?",
    "Are there circumstances in which original documents are required?",
    "What evidence can be used to support a claim?",
    "What documentation should be kept in case Aurora requests further evidence?",
    "How should supporting documents be provided when responding to a claim query?",
    "What happens when the documents submitted with a claim are incomplete?",

    "What documentation is relevant when a claim involves an implant?",
    "What evidence may be needed for pharmacy-related expenses?",
    "What supporting records are relevant to an accident-related claim?",
    "What information may be required from a hospital for reimbursement?",
    "What documentation can support an investigation-related expense?",
    "What should be submitted when a claim includes multiple types of medical expenses?",
    "What evidence may be requested when the submitted expenses need clarification?",
    "How should a claimant respond when Aurora asks for documentation concerning a particular expense?",

    "What is the grace period for policy renewal?",
    "What happens during the renewal grace period?",
    "What should a policyholder know before renewing the policy?",
    "What does the policy specify about renewal?",
    "What happens if renewal is not completed within the applicable period?",
    "Are there any conditions associated with the renewal grace period?",

    "When does Aurora require KYC documentation?",
    "What type of information is relevant for KYC verification?",
    "Why might KYC documentation be requested during the policy process?",
    "What should a customer do if Aurora requests KYC documents?",
    "What customer information can be obtained from the policy records?",
    "What information is needed to verify a customer's policy details?",

    "What is the grievance redressal process?",
    "When should a customer escalate an unresolved issue?",
    "What is the role of the grievance redressal officer?",
    "What timeframe applies to escalation through the grievance process?",

    "How do the claim submission deadline and the deadline for responding to an Aurora query differ?",
    "What are the main timing requirements a claimant should remember from submission through follow-up?",
    "If a claimant submits a reimbursement request after discharge and Aurora later asks for more information, what deadlines are relevant?",
    "What are the key steps a customer should follow when submitting and responding to a reimbursement claim?"
)



$targetUncached = 50
$results = @()

$currentQuestion = 0
$uncachedCount = 0

Write-Host ""
Write-Host "============================================================"
Write-Host "LAB 7 PERFORMANCE EXPERIMENT"
Write-Host "TARGET: 50 UNCACHED SAMPLES"
Write-Host "============================================================"
Write-Host ""

foreach ($question in $questions) {

    if ($uncachedCount -ge $targetUncached) {
        break
    }

    $currentQuestion++

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "Question bank item: $currentQuestion / $($questions.Count)"
    Write-Host "Uncached samples: $uncachedCount / $targetUncached"
    Write-Host "------------------------------------------------------------"
    Write-Host $question

    $body = @{
        question = $question
        top_k = 5
        mode = "rag"
    } | ConvertTo-Json

    try {

        $response = Invoke-RestMethod `
            -Uri "http://127.0.0.1:8000/ask" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body

        if ($response.cached -eq $false) {

            $uncachedCount++

            $results += [PSCustomObject]@{
                Sample       = $uncachedCount
                BankNumber   = $currentQuestion
                Question     = $question
                Latency_ms   = [double]$response.latency_ms
                Cost_usd     = [double]$response.cost_usd
                Cached       = $response.cached
                Refused      = $response.refused
                TraceId      = $response.trace_id
            }

            Write-Host "UNCACHED SAMPLE ACCEPTED"
            Write-Host "Sample  : $uncachedCount / $targetUncached"
            Write-Host "Latency : $($response.latency_ms) ms"
            Write-Host "Cost    : $($response.cost_usd)"
            Write-Host "Trace   : $($response.trace_id)"
        }
        else {

            Write-Host "CACHE HIT - NOT COUNTED"
            Write-Host "Latency : $($response.latency_ms) ms"
        }
    }
    catch {

        Write-Host "ERROR:"
        Write-Host $_.Exception.Message
    }
}

# ============================================================
# SAVE
# ============================================================

$outputFile = ".\reports\lab7_performance_50_uncached.csv"

$results |
    Export-Csv $outputFile -NoTypeInformation -Encoding UTF8

# ============================================================
# STATISTICS
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "FINAL EXPERIMENT RESULTS"
Write-Host "============================================================"

Write-Host "Question bank used : $currentQuestion"
Write-Host "Target samples     : $targetUncached"
Write-Host "Uncached samples   : $($results.Count)"

if ($results.Count -gt 0) {

    $latencies = @(
        $results |
        ForEach-Object { [double]$_.Latency_ms } |
        Sort-Object
    )

    $costs = @(
        $results |
        ForEach-Object { [double]$_.Cost_usd }
    )

    function Get-Percentile {
        param(
            [double[]]$Values,
            [double]$Percentile
        )

        $sorted = $Values | Sort-Object

        if ($sorted.Count -eq 1) {
            return $sorted[0]
        }

        $rank = ($Percentile / 100) * ($sorted.Count - 1)

        $lower = [math]::Floor($rank)
        $upper = [math]::Ceiling($rank)

        if ($lower -eq $upper) {
            return $sorted[$lower]
        }

        $weight = $rank - $lower

        return (
            $sorted[$lower] +
            (($sorted[$upper] - $sorted[$lower]) * $weight)
        )
    }

    $p50 = Get-Percentile $latencies 50
    $p90 = Get-Percentile $latencies 90
    $p95 = Get-Percentile $latencies 95
    $p99 = Get-Percentile $latencies 99

    $average = ($latencies | Measure-Object -Average).Average
    $minimum = ($latencies | Measure-Object -Minimum).Minimum
    $maximum = ($latencies | Measure-Object -Maximum).Maximum

    $totalCost = ($costs | Measure-Object -Sum).Sum
    $averageCost = ($costs | Measure-Object -Average).Average

    $above6s = @(
        $results |
        Where-Object { [double]$_.Latency_ms -gt 6000 }
    ).Count

    $within6s = $results.Count - $above6s

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "UNCACHED LATENCY"
    Write-Host "------------------------------------------------------------"

    Write-Host "Samples          : $($results.Count)"
    Write-Host "Minimum          : $([math]::Round($minimum,2)) ms"
    Write-Host "P50              : $([math]::Round($p50,2)) ms"
    Write-Host "P90              : $([math]::Round($p90,2)) ms"
    Write-Host "P95              : $([math]::Round($p95,2)) ms"
    Write-Host "P99              : $([math]::Round($p99,2)) ms"
    Write-Host "Maximum          : $([math]::Round($maximum,2)) ms"
    Write-Host "Average          : $([math]::Round($average,2)) ms"

    Write-Host ""
    Write-Host "Above 6000 ms    : $above6s"
    Write-Host "At/below 6000 ms: $within6s"

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "COST"
    Write-Host "------------------------------------------------------------"

    Write-Host "Total cost       : $([math]::Round($totalCost,6)) USD"
    Write-Host "Average/query    : $([math]::Round($averageCost,6)) USD"

    Write-Host ""
    Write-Host "------------------------------------------------------------"
    Write-Host "TARGETS"
    Write-Host "------------------------------------------------------------"

    Write-Host "P95 target       : 6000 ms"
    Write-Host "Cost target      : 0.010000 USD/query"

    if ($p95 -le 6000) {
        Write-Host "Latency          : PASS"
    }
    else {
        Write-Host "Latency          : FAIL"
    }

    if ($averageCost -le 0.01) {
        Write-Host "Cost             : PASS"
    }
    else {
        Write-Host "Cost             : FAIL"
    }
}

Write-Host ""
Write-Host "Results saved to:"
Write-Host $outputFile
Write-Host ""
Write-Host "Experiment complete."