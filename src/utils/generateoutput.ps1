param(
    [Parameter(Mandatory=$true)]
    [string]$XmlFileName
)
Write-Host "Launching script for... $XmlFileName"
# --- CONFIG ---
$originalLocation = Get-Location
$baseSourceFolder = "D:\dev\Quality-Diversity-for-Racing-Track-Design\data\voronoi\xmlTracks"
$sourceFile = Join-Path $baseSourceFolder $XmlFileName

$destFolder = "C:\Program Files (x86)\torcs\tracks\road\output"
$destFile = Join-Path $destFolder "output.xml"
$torcsFolder = "C:\Program Files (x86)\torcs"
$localOutputFolder = "C:\Users\milob\AppData\Local\VirtualStore\Program Files (x86)\torcs\tracks\road\output"

# --- COPY & RENAME ---
Write-Host "Copying XML track... $sourceFile to $destFile"
Copy-Item -Path $sourceFile -Destination $destFile -Force

# --- RUN trackgen ---
Write-Host "Running trackgen..."
Set-Location $torcsFolder
.\trackgen.exe -c road -n output

# move output.ac and output-trk.ac to local output folder
# Write-Host "Moving output files to local output folder..."
# $acFile = Join-Path $destFolder "output.ac"
# $trkFile = Join-Path $destFolder "output-trk.ac"
# Move-Item -Path $acFile -Destination $localOutputFolder -Force
# Move-Item -Path $trkFile -Destination $localOutputFolder -Force


# --- Go back to original folder ---
Set-Location $originalLocation
