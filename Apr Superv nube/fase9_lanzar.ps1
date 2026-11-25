# Lanza (o REANUDA tras un apagon) la fase 9: tres procesos ocultos que siguen
# vivos aunque se cierre esta ventana. Se puede ejecutar las veces que haga
# falta: lo ya medido no se repite y cada entrenamiento a medias continua desde
# su ultimo guardado.
$d = $PSScriptRoot
$s = Join-Path $d "datos\modelos\fase9"
$py = Join-Path (Split-Path $d -Parent) ".venv\Scripts\python.exe"

$vivos = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*fase9.py*' }
if ($vivos) { Write-Host "La fase 9 ya esta en marcha."; exit }

New-Item -ItemType Directory -Force $s | Out-Null
$sello = Get-Date -Format "MMdd_HHmmss"
foreach ($i in 0, 1, 2) {
    $a = @("-u", "fase9.py", "--idt", "$i", "--frac-vram", "0.27")
    # Solo el primero libera las reservas que dejo el apagon; los otros dos
    # arrancan cuando ya lo ha hecho.
    if ($i -eq 0) { $a += "--reanudar" }
    Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $d `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $s "log_t${i}_$sello.txt") `
        -RedirectStandardError (Join-Path $s "err_t${i}_$sello.txt")
    if ($i -eq 0) { Start-Sleep -Seconds 40 }
}
Write-Host "Fase 9 lanzada. Resultados: python fase9.py --resumen"
