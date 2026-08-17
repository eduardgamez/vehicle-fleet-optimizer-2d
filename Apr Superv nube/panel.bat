@echo off
rem Doble clic aqui para abrir el panel de control de la busqueda.
rem Abre el navegador en http://localhost:8770 con la grafica y los botones
rem de Arrancar y Parar. Cerrar esta ventana cierra el panel, pero NO para la
rem busqueda: para eso esta el boton Parar.
cd /d "%~dp0"
"%~dp0..\.venv\Scripts\python.exe" "%~dp0panel.py" %*
pause
