@echo off
rem Doble clic aqui para abrir el panel de la FASE 1 (el barrido al azar).
rem Abre el navegador en http://localhost:8770 con el avance y los botones de
rem Arrancar y Parar.
rem
rem Al arrancar retoma donde lo dejo: lo ya evaluado esta en
rem datos\modelos\barrido_t*.csv y no se repite.
rem
rem Cerrar esta ventana cierra el panel, pero NO para el barrido: para eso esta
rem el boton Parar.
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" "%~dp0panel_fase1.py" %*
pause
