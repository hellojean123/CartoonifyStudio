# Deploy only the explicitly listed Cartoonify Studio upgrade artifacts. No deletion.
$ErrorActionPreference = 'Stop'
$sourceRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
$targetRoot = 'D:\CartoonifyStudio'
if (-not (Test-Path -LiteralPath (Join-Path $targetRoot 'cartoonify\core.py'))) {
    throw 'Expected Cartoonify Studio project not found; refusing deployment.'
}
$files = @('app.py','evaluate.py','benchmark_v2.py','fetch_highres.py','smoke_test.py','validate_delivery.py','build_gallery.py','README.md','benchmark.md','upgrade_report.md','test_report.md','requirements.txt','requirements-lock.txt','pytest.ini','start.bat','deploy_upgrade.ps1',
    'cartoonify\core.py','cartoonify\cache.py','cartoonify\service.py','cartoonify\__init__.py','tests\test_core.py','tests\test_upgrade.py')
$files += @('benchmarks\__init__.py','benchmarks\legacy_v1.py','benchmarks\README.md','tests\test_publication.py','.gitignore')
$outputFiles = @('gallery.html','blind_review.html','blind_key.json','timings.json','quality_metrics.json','benchmark_v2.json','quantization_benchmark.json','pytest.xml','pytest_core.xml','smoke_test.json')
foreach ($name in $outputFiles) { $files += 'outputs\' + $name }
foreach ($directory in @('1_经典卡通','2_扁平插画','3_美式漫画','4_铅笔素描','upgrade_comparisons','blind')) {
    $folder = Join-Path $sourceRoot ('outputs\' + $directory)
    foreach ($file in (Get-ChildItem -LiteralPath $folder -File -Recurse)) { $files += $file.FullName.Substring($sourceRoot.Length + 1) }
}
foreach ($file in (Get-ChildItem -LiteralPath (Join-Path $sourceRoot 'outputs') -File)) {
    if ($file.Name -like '*_comparison.jpg' -or $file.Name -like 'highres_*.png') { $files += 'outputs\' + $file.Name }
}
$manifest = @()
foreach ($relative in ($files | Select-Object -Unique)) {
    $from = [System.IO.Path]::GetFullPath((Join-Path $sourceRoot $relative))
    $to = [System.IO.Path]::GetFullPath((Join-Path $targetRoot $relative))
    if (-not $from.StartsWith($sourceRoot + '\') -or -not $to.StartsWith($targetRoot + '\')) { throw 'Path escaped project' }
    if (-not (Test-Path -LiteralPath $from -PathType Leaf)) { throw ('Missing deliverable: ' + $relative) }
    $parent = Split-Path -Parent $to
    if (-not (Test-Path -LiteralPath $parent)) { New-Item -ItemType Directory -Path $parent -Force | Out-Null }
    if ($from -ne $to) { Copy-Item -LiteralPath $from -Destination $to -Force }
    $expected = (Get-FileHash -LiteralPath $from -Algorithm SHA256).Hash
    $actual = (Get-FileHash -LiteralPath $to -Algorithm SHA256).Hash
    if ($expected -ne $actual) { throw ('Hash mismatch: ' + $relative) }
    $manifest += [PSCustomObject]@{ file = $relative; sha256 = $actual }
}
$manifest | ConvertTo-Json -Depth 4 | Out-File -LiteralPath (Join-Path $targetRoot 'deployment_manifest.json') -Encoding utf8
Write-Output ('Deployed and SHA256 verified ' + $manifest.Count + ' files to ' + $targetRoot)
