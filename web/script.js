document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    source: 'aleatorio',
    optim: 'secuencial',
    quality: 3,
    playing: false,
    mundo: { W: 40, H: 24 },
    obstaculos: [],
    vehiculos: [],
    trayectorias: [],
    frames: [],
    frameIndex: 0,
    animId: null
  };

  // DOM Elements
  const qualitySlider = document.getElementById('quality-slider');
  const qualityFill = document.getElementById('quality-fill');
  const qualityThumb = document.getElementById('quality-thumb');
  const qualityVal = document.getElementById('quality-val');

  const segSource = document.getElementById('seg-source');
  const thumbSource = document.getElementById('thumb-source');
  const sourceButtons = segSource?.querySelectorAll('button') || [];

  const segOptim = document.getElementById('seg-optim');
  const thumbOptim = document.getElementById('thumb-optim');
  const optimButtons = segOptim?.querySelectorAll('button') || [];

  const btnPlay = document.getElementById('btn-play');
  const iconPlay = document.getElementById('icon-play');
  const iconPause = document.getElementById('icon-pause');
  const btnRestart = document.getElementById('btn-restart');
  const btnReproducir = document.getElementById('btn-reproducir');

  const btnInsertar = document.getElementById('btn-insertar');
  const btnSimular = document.getElementById('btn-simular');
  const btnNuevasRutas = document.getElementById('btn-nuevas-rutas');

  const btnMapaAleatorio = document.getElementById('btn-mapa-aleatorio');
  const btnMapaImportar = document.getElementById('btn-mapa-importar');
  const btnMapaGuardar = document.getElementById('btn-mapa-guardar');
  const btnMapaCargar = document.getElementById('btn-mapa-cargar');
  const fileImportar = document.getElementById('file-importar');
  const fileCargar = document.getElementById('file-cargar');

  const numVehiculosInput = document.getElementById('num-vehiculos');
  const densidadInput = document.getElementById('densidad-obs');
  const maxCandInput = document.getElementById('max-cand');
  const candTotalEl = document.getElementById('cand-total');
  const vehiculosInput = document.getElementById('vehiculos-input');
  const statusText = document.getElementById('status-text');
  const statusSpinner = document.getElementById('status-spinner');
  const setLoading = (loading) => {
    if (statusSpinner) statusSpinner.hidden = !loading;
  };

  // --- Total de combinaciones («de X») ---
  // Número TEÓRICO de órdenes que el modo podría llegar a probar (sin el tope de
  // «Órdenes a explorar»): global = n!; grupos = producto de factoriales del
  // tamaño de cada bucket (grupo, prioridad); secuencial = 1. Se calcula a partir
  // de los vehículos ya construidos (llevan grupo/prioridad), que es donde las
  // preferencias quedan definidas — igual en aleatorio (tras generarlos) que en
  // manual (tras insertarlos).
  const factorial = (n) => {
    let r = 1;
    for (let i = 2; i <= n; i++) r *= i;
    return r;
  };

  const totalCombos = (vehiculos, modo) => {
    const n = vehiculos.length;
    if (n === 0) return null;
    if (modo === 'secuencial') return 1;
    if (modo === 'global') return factorial(n);
    // 'prioridades' (grupos): producto de factoriales por bucket (grupo|prioridad).
    const buckets = {};
    vehiculos.forEach((v) => {
      const k = `${v.grupo ?? 1}|${v.prioridad ?? 1}`;
      buckets[k] = (buckets[k] || 0) + 1;
    });
    return Object.values(buckets).reduce((acc, s) => acc * factorial(s), 1);
  };

  const updateCombos = () => {
    if (!candTotalEl) return;
    const total = totalCombos(state.vehiculos || [], state.optim);
    candTotalEl.textContent = total === null
      ? 'de —'
      : `de ${total.toLocaleString('es-ES')}`;
  };

  const canvas = document.getElementById('main-canvas');
  const ctx = canvas ? canvas.getContext('2d') : null;

  // Setup Canvas para agudeza Retina
  const setupCanvas = () => {
    if (!canvas || !ctx) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawScene();
  };

  window.addEventListener('resize', setupCanvas);

  // --- Quality Slider Logic ---
  const updateQuality = (pct) => {
    const val = Math.round(1 + pct * 4);
    state.quality = val;
    if (qualityVal) qualityVal.textContent = val;
    const frac = (val - 1) / 4;
    if (qualityFill) qualityFill.style.width = (frac * 100) + '%';
    if (qualityThumb) {
      qualityThumb.style.left = `calc((100% - var(--thumb-size)) * ${frac})`;
    }
  };

  const handlePointer = (e) => {
    if (e.type === 'pointermove' && e.buttons !== 1) return;
    const r = qualitySlider.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    updateQuality(p);
  };

  if (qualitySlider) {
    qualitySlider.addEventListener('pointerdown', handlePointer);
    qualitySlider.addEventListener('pointermove', handlePointer);
  }

  // --- Segmented Control: Source ---
  sourceButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
      sourceButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.source = btn.getAttribute('data-val');
      if (thumbSource) {
        thumbSource.style.left = `calc(${index} * 100% / ${sourceButtons.length} + 2px)`;
      }
      // Al pasar a ALEATORIO se generan las instrucciones (rellena la caja y
      // define las preferencias) para que el total sea calculable; en manual el
      // total refleja los vehículos ya insertados.
      if (state.source === 'aleatorio') {
        generarVehiculos();
      } else {
        updateCombos();
      }
    });
  });

  // --- Segmented Control: Optim ---
  optimButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
      optimButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.optim = btn.getAttribute('data-val');
      if (thumbOptim) {
        thumbOptim.style.left = `calc(${index} * 100% / ${optimButtons.length} + 2px)`;
      }
      updateCombos();          // el total depende del modo elegido
    });
  });

  // --- Dibujo en el Canvas ---
  const drawScene = () => {
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;
    ctx.clearRect(0, 0, w, h);

    // Fondo claro como en multi_v_evo.py (#f4f6fb)
    ctx.fillStyle = '#f4f6fb';
    ctx.fillRect(0, 0, w, h);

    const scale = Math.min(w / state.mundo.W, h / state.mundo.H);
    const offsetX = (w - state.mundo.W * scale) / 2;
    const offsetY = (h - state.mundo.H * scale) / 2;

    // Dibujar obstáculos (#3a4252 con borde #1f2530)
    ctx.fillStyle = '#3a4252';
    ctx.strokeStyle = '#1f2530';
    ctx.lineWidth = 1;
    state.obstaculos.forEach(pol => {
      if (pol.length < 3) return;
      ctx.beginPath();
      ctx.moveTo(pol[0][0] * scale + offsetX, pol[0][1] * scale + offsetY);
      for (let i = 1; i < pol.length; i++) {
        ctx.lineTo(pol[i][0] * scale + offsetX, pol[i][1] * scale + offsetY);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    });

    // Dibujar trayectorias completas si están calculadas
    state.vehiculos.forEach(v => {
      const color = v.color || '#007aff';
      const tInfo = state.trayectorias.find(t => String(t.id) === String(v.id));
      if (tInfo && tInfo.traj && tInfo.traj.length >= 2) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        const pt0 = tInfo.traj[0];
        ctx.moveTo(pt0[0] * scale + offsetX, pt0[1] * scale + offsetY);
        for (let i = 1; i < tInfo.traj.length; i++) {
          const pt = tInfo.traj[i];
          ctx.lineTo(pt[0] * scale + offsetX, pt[1] * scale + offsetY);
        }
        ctx.stroke();
      }
    });

    // Dibujar vehículos / metas
    state.vehiculos.forEach(v => {
      const color = v.color || '#007aff';
      const [ix, iy] = v.inicio;
      const [mx, my] = v.meta;

      // Círculo de meta
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(mx * scale + offsetX, my * scale + offsetY, 4.5, 0, Math.PI * 2);
      ctx.stroke();

      // Flecha de ángulo de llegada si existe
      if (v.meta_th !== null && v.meta_th !== undefined) {
        const ax = mx + 0.3 * Math.cos(v.meta_th);
        const ay = my + 0.3 * Math.sin(v.meta_th);
        const axpx = ax * scale + offsetX;
        const aypx = ay * scale + offsetY;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(mx * scale + offsetX, my * scale + offsetY);
        ctx.lineTo(axpx, aypx);
        ctx.stroke();

        const headlen = 2;
        const angle = v.meta_th;
        ctx.beginPath();
        ctx.moveTo(axpx, aypx);
        ctx.lineTo(
          axpx - headlen * Math.cos(angle - Math.PI / 6),
          aypx - headlen * Math.sin(angle - Math.PI / 6)
        );
        ctx.moveTo(axpx, aypx);
        ctx.lineTo(
          axpx - headlen * Math.cos(angle + Math.PI / 6),
          aypx - headlen * Math.sin(angle + Math.PI / 6)
        );
        ctx.stroke();
      }

      // Pose actual del vehículo
      let currX = ix;
      let currY = iy;
      let currTh = (v.inicio.length > 2) ? v.inicio[2] : 0;

      if (state.frames && state.frames.length > 0) {
        const fIdx = Math.min(state.frameIndex, state.frames.length - 1);
        const fr = state.frames[fIdx];
        const pose = fr ? fr[String(v.id)] : null;
        if (pose) {
          currX = pose[0];
          currY = pose[1];
          currTh = pose[2];
        }
      }

      const lPx = Math.max(16, (v.largo || 1.3) * scale);
      const wPx = Math.max(9, (v.ancho || 0.7) * scale);

      // Rectángulo de vehículo rotado
      ctx.save();
      ctx.translate(currX * scale + offsetX, currY * scale + offsetY);
      ctx.rotate(currTh);

      ctx.fillStyle = color;
      ctx.strokeStyle = '#101010';
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(-lPx / 2, -wPx / 2, lPx, wPx, 3);
      } else {
        ctx.rect(-lPx / 2, -wPx / 2, lPx, wPx);
      }
      ctx.fill();
      ctx.stroke();

      // Indicador de dirección: trazo corto con la punta pegada al borde
      // frontal del rectángulo. No arranca del centro para no cruzar por
      // encima del número del vehículo y dejarlo legible.
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(lPx * 0.32, 0);
      ctx.lineTo(lPx * 0.5, 0);
      ctx.stroke();
      ctx.restore();

      // ID del vehículo
      ctx.save();
      ctx.translate(currX * scale + offsetX, currY * scale + offsetY);
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 11px -apple-system, system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(v.id), 0, 0);
      ctx.restore();
    });
  };

  // --- Bucle de Animación ---
  // Un ÚNICO bucle de requestAnimationFrame a la vez. Arrancar la reproducción
  // sin cancelar antes el bucle anterior dejaba dos (o más) bucles corriendo en
  // paralelo, y cada uno avanzaba el fotograma una vez por refresco → el doble
  // de velocidad ("cámara rápida"). Todo arranque pasa por iniciarReproduccion,
  // que primero cancela cualquier bucle vivo, de modo que nunca se solapan.
  const mostrarPausa = () => {
    if (iconPlay) iconPlay.classList.add('icon-hidden');
    if (iconPause) iconPause.classList.remove('icon-hidden');
  };
  const mostrarPlay = () => {
    if (iconPause) iconPause.classList.add('icon-hidden');
    if (iconPlay) iconPlay.classList.remove('icon-hidden');
  };

  const detenerAnim = () => {
    if (state.animId) {
      cancelAnimationFrame(state.animId);
      state.animId = null;
    }
  };

  const animStep = () => {
    if (!state.playing) return;
    drawScene();

    if (state.frames && state.frameIndex < state.frames.length - 1) {
      state.frameIndex++;
      state.animId = requestAnimationFrame(animStep);
    } else {
      state.playing = false;
      state.animId = null;
      mostrarPlay();
    }
  };

  const iniciarReproduccion = (desdeInicio) => {
    detenerAnim();                       // garantiza un solo bucle activo
    if (desdeInicio) state.frameIndex = 0;
    state.playing = true;
    mostrarPausa();
    state.animId = requestAnimationFrame(animStep);
  };

  btnPlay?.addEventListener('click', () => {
    if (state.playing) {
      state.playing = false;
      detenerAnim();
      mostrarPlay();
    } else {
      // Si ya estaba al final, reanudar equivale a reproducir desde el inicio.
      const alFinal = state.frames && state.frameIndex >= state.frames.length - 1;
      iniciarReproduccion(alFinal);
    }
  });

  btnRestart?.addEventListener('click', () => {
    state.playing = false;
    detenerAnim();
    mostrarPlay();
    state.frameIndex = 0;
    drawScene();
  });

  btnReproducir?.addEventListener('click', () => {
    iniciarReproduccion(true);
  });

  // --- Inicialización del Mapa y Estado ---
  const initApp = async () => {
    try {
      const res = await fetch('/api/init');
      const data = await leerRespuesta(res);
      if (data.ok) {
        state.obstaculos = data.obstaculos || [];
        state.vehiculos = data.vehiculos || [];
        if (data.mundo) state.mundo = data.mundo;
        if (data.texto && vehiculosInput) {
          vehiculosInput.value = data.texto;
        }
        statusText.textContent = "Listo. Mapa cargado. Puedes insertar o generar vehículos.";
        updateCombos();
        setupCanvas();
      }
    } catch (err) {
      statusText.textContent = `Error cargando mapa: ${err.message}`;
    }
  };

  // --- Conexión con Endpoints API (/api/generar) ---
  // Lee la respuesta sin asumir que es JSON: si el servidor devuelve una
  // página HTML de error (502/504 del proxy de Render, o un 500), muestra el
  // motivo real en vez de "The string did not match the expected pattern".
  const leerRespuesta = async (res) => {
    const texto = await res.text();
    try {
      return JSON.parse(texto);
    } catch (_) {
      const motivo = res.status === 502 || res.status === 503 || res.status === 504
        ? `El servidor no respondió (código ${res.status}). Puede que la simulación consumiera demasiada memoria o tardara demasiado. Prueba con menos vehículos o menor calidad.`
        : `El servidor devolvió una respuesta inesperada (código ${res.status}).`;
      return { ok: false, error: motivo };
    }
  };

  const generarVehiculos = async () => {
    statusText.textContent = "Generando vehículos en servidor...";
    try {
      const res = await fetch('/api/generar', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modo: state.source,
          num_vehiculos: parseInt(numVehiculosInput?.value || 4),
          texto: vehiculosInput?.value || ""
        })
      });
      const data = await leerRespuesta(res);
      if (!data.ok) {
        statusText.textContent = `Error: ${data.error}`;
        return;
      }

      state.obstaculos = data.obstaculos || state.obstaculos;
      state.vehiculos = data.vehiculos || [];
      state.trayectorias = [];
      state.frames = [];
      state.frameIndex = 0;
      if (data.mundo) state.mundo = data.mundo;
      // Reescribe la caja de texto con la definición de los vehículos recién
      // generados (o con el texto normalizado por el servidor en modo manual),
      // para que refleje siempre el estado real y sea editable/reutilizable.
      if (data.texto && vehiculosInput) {
        vehiculosInput.value = data.texto;
      }

      const sufijo = state.source === 'aleatorio'
        ? ' La caja de texto muestra sus datos.'
        : '';
      statusText.textContent =
        `${state.vehiculos.length} vehículos cargados/generados. Pulsa «Simular».${sufijo}`;
      updateCombos();
      drawScene();
    } catch (err) {
      statusText.textContent = `Error de red: ${err.message}`;
    }
  };

  btnInsertar?.addEventListener('click', generarVehiculos);

  // Cambiar el nº de vehículos en modo ALEATORIO regenera las instrucciones (y,
  // con ellas, el total de combinaciones). 'change' salta al perder foco o con
  // Enter, no en cada tecla.
  numVehiculosInput?.addEventListener('change', () => {
    if (state.source === 'aleatorio') generarVehiculos();
  });

  // --- Botones de Mapa ---
  btnMapaAleatorio?.addEventListener('click', async () => {
    statusText.textContent = "Generando nuevo mapa aleatorio...";
    try {
      const res = await fetch('/api/mapa/aleatorio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          densidad: parseFloat(densidadInput?.value || 0.0)
        })
      });
      const data = await leerRespuesta(res);
      if (data.ok) {
        state.obstaculos = data.obstaculos || [];
        state.vehiculos = [];
        state.trayectorias = [];
        state.frames = [];
        state.frameIndex = 0;
        statusText.textContent = "Nuevo mapa aleatorio generado. Inserta vehículos.";
        drawScene();
      }
    } catch (err) {
      statusText.textContent = `Error al generar mapa: ${err.message}`;
    }
  });

  btnMapaImportar?.addEventListener('click', () => {
    fileImportar?.click();
  });

  fileImportar?.addEventListener('change', async () => {
    if (!fileImportar.files || fileImportar.files.length === 0) return;
    const formData = new FormData();
    formData.append('archivo', fileImportar.files[0]);
    statusText.textContent = "Importando imagen como mapa...";
    try {
      const res = await fetch('/api/mapa/importar', {
        method: 'POST',
        body: formData
      });
      const data = await leerRespuesta(res);
      if (data.ok) {
        state.obstaculos = data.obstaculos || [];
        state.vehiculos = [];
        state.trayectorias = [];
        state.frames = [];
        state.frameIndex = 0;
        statusText.textContent = "Mapa importado desde imagen. Inserta vehículos.";
        drawScene();
      } else {
        statusText.textContent = `Error importando imagen: ${data.error}`;
      }
    } catch (err) {
      statusText.textContent = `Error: ${err.message}`;
    }
  });

  btnMapaGuardar?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/mapa/guardar');
      const data = await leerRespuesta(res);
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'mapa_guardado.json';
      a.click();
      URL.revokeObjectURL(url);
      statusText.textContent = "Mapa guardado como archivo JSON.";
    } catch (err) {
      statusText.textContent = `Error al guardar mapa: ${err.message}`;
    }
  });

  btnMapaCargar?.addEventListener('click', () => {
    fileCargar?.click();
  });

  fileCargar?.addEventListener('change', async () => {
    if (!fileCargar.files || fileCargar.files.length === 0) return;
    const formData = new FormData();
    formData.append('archivo', fileCargar.files[0]);
    statusText.textContent = "Cargando mapa desde archivo...";
    try {
      const res = await fetch('/api/mapa/cargar', {
        method: 'POST',
        body: formData
      });
      const data = await leerRespuesta(res);
      if (data.ok) {
        state.obstaculos = data.obstaculos || [];
        state.vehiculos = [];
        state.trayectorias = [];
        state.frames = [];
        state.frameIndex = 0;
        statusText.textContent = "Mapa cargado correctamente. Inserta vehículos.";
        drawScene();
      } else {
        statusText.textContent = `Error cargando mapa: ${data.error}`;
      }
    } catch (err) {
      statusText.textContent = `Error: ${err.message}`;
    }
  });

  // El cálculo ya NO bloquea la petición: el servidor lo corre en segundo plano
  // y aquí se consulta el progreso hasta que termina. Así el navegador nunca
  // corta por timeout ("Load failed"), por muy difícil que sea la planificación.
  const dormir = (ms) => new Promise((r) => setTimeout(r, ms));

  const ejecutarConProgreso = async (url, body) => {
    // 1) Lanza el trabajo (respuesta inmediata).
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const arranque = await leerRespuesta(res);
    if (!res.ok || !arranque.ok) {
      throw new Error(arranque.error || 'No se pudo iniciar la simulación.');
    }
    // 2) Sondea el estado y va mostrando el progreso hasta done/error.
    while (true) {
      await dormir(700);
      let est;
      try {
        const r = await fetch(url + '/estado');
        est = await leerRespuesta(r);
      } catch (_) {
        continue;   // fallo puntual de red al sondear: reintenta
      }
      if (est.progreso) statusText.textContent = est.progreso;
      if (est.estado === 'done') return est.resultado;
      if (est.estado === 'error') throw new Error(est.error || 'Error en el cálculo.');
    }
  };

  // --- Conexión con Endpoints API (/api/simular) ---
  const simularFlota = async () => {
    if (state.vehiculos.length === 0) {
      await generarVehiculos();
    }
    statusText.textContent = "Calculando rutas optimizadas en Python...";
    setLoading(true);
    try {
      const data = await ejecutarConProgreso('/api/simular', {
        calidad: state.quality,
        optimizacion: state.optim,
        max_cand: parseInt(maxCandInput?.value || 12)
      });
      if (!data || !data.ok) {
        statusText.textContent = `Error en simulación: ${data && data.error ? data.error : 'sin resultado'}`;
        return;
      }
      state.trayectorias = data.trayectorias || [];
      state.frames = data.frames || [];
      statusText.textContent = "¡Simulación en curso!";
      iniciarReproduccion(true);
    } catch (err) {
      statusText.textContent = `Error de simulación: ${err.message}`;
    } finally {
      setLoading(false);
    }
  };

  btnSimular?.addEventListener('click', simularFlota);

  // --- Nuevas Rutas ---
  btnNuevasRutas?.addEventListener('click', async () => {
    statusText.textContent = "Calculando nuevas rutas para vehículos del texto...";
    setLoading(true);
    try {
      const data = await ejecutarConProgreso('/api/nuevas_rutas', {
        texto: vehiculosInput?.value || "",
        calidad: state.quality,
        optimizacion: state.optim,
        max_cand: parseInt(maxCandInput?.value || 12)
      });
      if (!data || !data.ok) {
        statusText.textContent = `Error: ${data && data.error ? data.error : 'sin resultado'}`;
        return;
      }
      state.vehiculos = data.vehiculos || state.vehiculos;
      state.trayectorias = data.trayectorias || [];
      state.frames = data.frames || [];
      updateCombos();
      statusText.textContent = "¡Simulación de nuevas rutas en curso!";
      iniciarReproduccion(true);
    } catch (err) {
      statusText.textContent = `Error: ${err.message}`;
    } finally {
      setLoading(false);
    }
  });

  // --- Barra de acceso rápido (móvil): reutilizan las mismas acciones ---
  document.getElementById('q-insertar')?.addEventListener('click', generarVehiculos);
  document.getElementById('q-simular')?.addEventListener('click', simularFlota);
  document.getElementById('q-reproducir')?.addEventListener('click', () => {
    if (state.frames && state.frames.length > 0) {
      iniciarReproduccion(true);
    } else {
      simularFlota();
    }
  });

  // Inicializar al cargar el documento
  initApp();
});
