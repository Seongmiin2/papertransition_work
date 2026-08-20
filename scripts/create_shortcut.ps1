$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$electron = Join-Path $projectRoot "node_modules\electron\dist\electron.exe"
if (-not (Test-Path -LiteralPath $electron)) {
    throw "Electron이 설치되지 않았습니다. 프로젝트 폴더에서 npm install을 먼저 실행하세요."
}
$desktop = [Environment]::GetFolderPath("Desktop")
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut((Join-Path $desktop "Exam2HWPX.lnk"))
$shortcut.TargetPath = $electron
$shortcut.Arguments = '"' + $projectRoot + '"'
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = $electron + ",0"
$shortcut.Description = "시험지 PDF를 편집 가능한 HWPX로 변환"
$shortcut.Save()
Write-Host "바탕화면에 Exam2HWPX.lnk를 만들었습니다."
