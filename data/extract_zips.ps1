# Define the starting directory
$startPath = "C:\Users\leond\Documents\03 FILES\Bluesky Data"

# Get all tar.gz files
$tarGzFiles = Get-ChildItem -Path $startPath -Recurse -Filter "*.tar.gz"

# Loop through each file
foreach ($file in $tarGzFiles) {
    Write-Host "Extracting $($file.FullName)..."

    # Create a destination folder with the same name as the file
    $destinationPath = $file.FullName.Replace($file.Extension, "").Replace(".tar", "")

    if (-not (Test-Path $destinationPath)) {
        New-Item -ItemType Directory -Path $destinationPath
    }

    # Use the native 'tar' command to extract the files
    # -x: eXtract, -z: gZip, -f: file, -C: Change directory
    try {
        tar -xzf $file.FullName -C $destinationPath
        Write-Host "Successfully extracted to $destinationPath" -ForegroundColor Green

    } catch {
        Write-Host "Failed to extract $($file.FullName). Error: $_" -ForegroundColor Red
    }
}

Write-Host "Extraction process complete."