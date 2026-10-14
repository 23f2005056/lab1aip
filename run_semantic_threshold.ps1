$thresholds = @(0.70, 0.75, 0.80, 0.85, 0.90, 0.95)

$questions = @(
    "What is the deadline for submitting a reimbursement claim?",
    "How much time does a claimant have to file the reimbursement request?",
    "When must a reimbursement claim be submitted after discharge?",
    "How long after hospital discharge can I submit my claim?",
    "How many days are allowed to respond to an Aurora query?",
    "What is the response period when Aurora requests additional information?",
    "When must post-hospitalisation expenses be submitted?",
    "What is the deadline for submitting bills after hospitalisation?",
    "What is the renewal grace period?",
    "How long is the grace period for policy renewal?",
    "When does Aurora require KYC documents?",
    "Under what circumstances is KYC documentation required?",
    "What is the grievance redressal process?",
    "How can a customer escalate an unresolved grievance?"
)

Write-Host ""
Write-Host "==============================================="
Write-Host "LAB 7 SEMANTIC CACHE THRESHOLD EXPERIMENT"
Write-Host "==============================================="

foreach ($threshold in $thresholds) {

    Write-Host ""
    Write-Host "==============================================="
    Write-Host "THRESHOLD: $threshold"
    Write-Host "==============================================="

    $env:SEMANTIC_THRESHOLD = "$threshold"

    Write-Host ""
    Write-Host "Start server with:"
    Write-Host "`$env:SEMANTIC_THRESHOLD='$threshold'"
    Write-Host "python -m uvicorn labs.lab7.service:app --host 127.0.0.1 --port 8000"
    Write-Host ""

    Read-Host "Press ENTER after the server is running"

    # Verify threshold
    $health = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/health" `
        -Method Get

    Write-Host ""
    Write-Host "Server threshold : $($health.semantic_threshold)"
    Write-Host "Semantic entries : $($health.semantic_cache_entries)"

    if ([double]$health.semantic_threshold -ne [double]$threshold) {
        Write-Host ""
        Write-Host "ERROR: Threshold mismatch."
        Write-Host "Expected: $threshold"
        Write-Host "Actual  : $($health.semantic_threshold)"
        exit 1
    }

    # Get starting metrics
    $before = Invoke-RestMethod `
        -Uri "http://127.0.0.1:8000/metrics" `
        -Method Get

    $startExact = [int]$before.cache.exact_hits
    $startSemantic = [int]$before.cache.semantic_hits
    $startMisses = [int]$before.cache.misses

    Write-Host ""
    Write-Host "Starting exact hits    : $startExact"
    Write-Host "Starting semantic hits : $startSemantic"
    Write-Host "Starting misses        : $startMisses"

    $results = @()

    foreach ($q in $questions) {

        $body = @{
            question = $q
            top_k = 5
            mode = "rag"
        } | ConvertTo-Json -Compress

        try {

            $response = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8000/ask" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body

            # Read metrics immediately after request
            $after = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8000/metrics" `
                -Method Get

            $exactNow = [int]$after.cache.exact_hits
            $semanticNow = [int]$after.cache.semantic_hits
            $missNow = [int]$after.cache.misses

            $exactDelta = $exactNow - $startExact
            $semanticDelta = $semanticNow - $startSemantic
            $missDelta = $missNow - $startMisses

            $cacheType = "none"

            if ($exactDelta -gt 0) {
                $cacheType = "exact"
                $startExact = $exactNow
            }
            elseif ($semanticDelta -gt 0) {
                $cacheType = "semantic"
                $startSemantic = $semanticNow
            }
            elseif ($missDelta -gt 0) {
                $cacheType = "miss"
                $startMisses = $missNow
            }

            $results += [PSCustomObject]@{
                Threshold = $threshold
                Question = $q
                Cached = $response.cached
                CacheType = $cacheType
                LatencyMs = [math]::Round($response.latency_ms, 2)
                Answer = ($response.answer -replace "`r|`n", " ")
            }

            Write-Host ""
            Write-Host "Q: $q"
            Write-Host "Cache type: $cacheType"
            Write-Host "Cached    : $($response.cached)"
            Write-Host "Latency   : $([math]::Round($response.latency_ms,2)) ms"

        }
        catch {
            Write-Host ""
            Write-Host "ERROR:"
            Write-Host $_.Exception.Message
        }
    }

    $exactCount = @(
        $results | Where-Object { $_.CacheType -eq "exact" }
    ).Count

    $semanticCount = @(
        $results | Where-Object { $_.CacheType -eq "semantic" }
    ).Count

    $missCount = @(
        $results | Where-Object { $_.CacheType -eq "miss" }
    ).Count

    Write-Host ""
    Write-Host "-----------------------------------------------"
    Write-Host "SUMMARY: THRESHOLD $threshold"
    Write-Host "-----------------------------------------------"
    Write-Host "Total questions : $($results.Count)"
    Write-Host "Exact hits      : $exactCount"
    Write-Host "Semantic hits   : $semanticCount"
    Write-Host "Misses          : $missCount"

    Write-Host ""
    $results | Format-Table CacheType, Cached, LatencyMs, Question -AutoSize

    $safeThreshold = $threshold.ToString().Replace(".", "_")

    $outputFile = ".\reports\semantic_threshold_$safeThreshold.csv"

    $results | Export-Csv `
        -Path $outputFile `
        -NoTypeInformation `
        -Encoding UTF8

    Write-Host ""
    Write-Host "Saved:"
    Write-Host $outputFile

    if ($threshold -ne $thresholds[-1]) {
        Write-Host ""
        Write-Host "STOP SERVER WITH Ctrl+C."
        Write-Host "Then restart it using the next threshold."
        Write-Host ""

        Read-Host "Press ENTER after restarting the server"
    }
}

Write-Host ""
Write-Host "==============================================="
Write-Host "EXPERIMENT COMPLETE"
Write-Host "==============================================="