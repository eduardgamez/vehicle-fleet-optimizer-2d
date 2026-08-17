# Barrido aleatorio (fase 1) en CASCADA, para que ninguna red se quede sin
# probar por tamaño.
#
# El problema: con varios procesos a la vez hay que repartir la memoria de la
# tarjeta, y las redes más grandes no caben en su parte. Si se bajan los
# procesos para que quepan, se desaprovecha la tarjeta durante horas con las
# pequeñas, que son la mayoría.
#
# La solución es hacerlo en pasadas, de más procesos a menos:
#   1) 3 procesos con un cuarto de tarjeta cada uno  → despacha la mayoría
#   2) 2 procesos con casi la mitad                  → recoge las que no cupieron
#   3) 1 proceso con la tarjeta entera               → las que aún no quepan
#
# Funciona porque una configuración que no cabe NO se anota como evaluada: queda
# pendiente y la recoge la pasada siguiente. Y las ya hechas no se repiten, así
# que cada pasada solo trabaja en lo que falta.
#
# Uso:  .\barrido_cascada.ps1 -NConfigs 240

param(
    [int]$NConfigs = 240,
    [int]$MaxIntentos = 40
)

$raiz = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path (Split-Path -Parent $raiz) ".venv\Scripts\python.exe"
$sc = Join-Path $raiz "entrenar_nube.py"

# El reparto en TAREAS es siempre 3 (cada una con su registro y su lista fija de
# configuraciones). Lo que cambia entre pasadas es cuántas se ejecutan A LA VEZ:
# de eso depende cuánta memoria le toca a cada una. Ojo con confundirlo: lanzar
# las 3 tareas pidiendo media tarjeta cada una suma más de lo que hay, y eso es
# lo que tumbaba el driver.
$TAREAS = 3

function Pasada($a_la_vez, $frac, $nombre) {
    Write-Host "`n=== $nombre : de $a_la_vez en $a_la_vez, $frac de tarjeta cada una ==="
    for ($base = 0; $base -lt $TAREAS; $base += $a_la_vez) {
    $trabajos = @()
    for ($i = $base; $i -lt [Math]::Min($base + $a_la_vez, $TAREAS); $i++) {
        $trabajos += Start-Job -ArgumentList $py, $sc, $i, $TAREAS, $NConfigs, $frac, $MaxIntentos -ScriptBlock {
            param($py, $sc, $idt, $total, $n, $frac, $maxIntentos)
            for ($k = 1; $k -le $maxIntentos; $k++) {
                & $py $sc --n-configs $n --semilla 0 --tarea $idt --tareas $total `
                    --criba="" --n-escenarios 400 --frac-vram $frac
                if ($LASTEXITCODE -eq 0) { break }
                "[t$idt] cayo (codigo $LASTEXITCODE), reintento $k"
            }
        }
    }
    $trabajos | Receive-Job -Wait
    $trabajos | Remove-Job -Force
    }
}

Pasada 3 0.27 "Pasada 1 - la mayoria, la tarjeta llena"
Pasada 2 0.44 "Pasada 2 - las que no cupieron"
Pasada 1 0.90 "Pasada 3 - las mas grandes, una a una"

Write-Host "`nBarrido completo. Registros en datos\modelos\barrido_t*.csv"
