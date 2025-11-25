# --- CONFIG ---
$sourceFile = "D:\dev\Quality-Diversity-for-Racing-Track-Design\data\voronoi\xmlTracks\output_100.xml"
$destFolder = "C:\Program Files (x86)\torcs\tracks\road\output"
$destFile = Join-Path $destFolder "output.xml"
$torcsFolder = "C:\Program Files (x86)\torcs"

# --- COPY & RENAME ---
Write-Host "Copying XML track... $sourceFile to $destFile"
Copy-Item -Path $sourceFile -Destination $destFile -Force

# --- RUN trackgen ---
Write-Host "Running trackgen..."
Set-Location $torcsFolder
.\trackgen.exe -c road -n output

# --- Go back to original folder ---
Set-Location -Path $PSScriptRoot
