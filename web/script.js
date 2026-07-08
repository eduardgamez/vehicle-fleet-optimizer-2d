document.addEventListener('DOMContentLoaded', () => {
  // State
  const state = {
    source: 'aleatorio',
    optim: 'secuencial',
    quality: 3,
    playing: false
  };

  // DOM Elements
  const qualitySlider = document.getElementById('quality-slider');
  const qualityFill = document.getElementById('quality-fill');
  const qualityThumb = document.getElementById('quality-thumb');
  const qualityVal = document.getElementById('quality-val');

  const segSource = document.getElementById('seg-source');
  const thumbSource = document.getElementById('thumb-source');
  const sourceButtons = segSource.querySelectorAll('button');

  const segOptim = document.getElementById('seg-optim');
  const thumbOptim = document.getElementById('thumb-optim');
  const optimButtons = segOptim.querySelectorAll('button');

  const btnPlay = document.getElementById('btn-play');
  const iconPlay = document.getElementById('icon-play');
  const iconPause = document.getElementById('icon-pause');

  // --- Quality Slider Logic ---
  const updateQuality = (pct) => {
    // 0 to 1
    const val = Math.round(1 + pct * 4);
    state.quality = val;
    qualityVal.textContent = val;
    
    // Update visual percentage (0% to 100%)
    const pctString = ((val - 1) / 4 * 100) + '%';
    qualityFill.style.width = pctString;
    qualityThumb.style.left = pctString;
  };

  const handlePointer = (e) => {
    if (e.type === 'pointermove' && e.buttons !== 1) return;
    const r = qualitySlider.getBoundingClientRect();
    const p = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    updateQuality(p);
  };

  qualitySlider.addEventListener('pointerdown', handlePointer);
  qualitySlider.addEventListener('pointermove', handlePointer);

  // --- Segmented Control: Source ---
  sourceButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
      sourceButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.source = btn.getAttribute('data-val');
      
      const widthPct = 100 / sourceButtons.length;
      thumbSource.style.left = `calc(${index} * 100% / ${sourceButtons.length} + 2px)`;
    });
  });

  // --- Segmented Control: Optim ---
  optimButtons.forEach((btn, index) => {
    btn.addEventListener('click', () => {
      optimButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.optim = btn.getAttribute('data-val');
      
      const widthPct = 100 / optimButtons.length;
      thumbOptim.style.left = `calc(${index} * 100% / ${optimButtons.length} + 2px)`;
    });
  });

  // --- Play / Pause ---
  btnPlay.addEventListener('click', () => {
    state.playing = !state.playing;
    if (state.playing) {
      iconPlay.classList.add('icon-hidden');
      iconPause.classList.remove('icon-hidden');
    } else {
      iconPause.classList.add('icon-hidden');
      iconPlay.classList.remove('icon-hidden');
    }
  });

});
