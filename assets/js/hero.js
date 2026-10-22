/* Hero background: one forward pass of AWARE, drawn as the Action Entropy field of Figure 1b (layers bottom to
   top, the 56 OpenVLA-OFT action elements left to right, viridis colour). A scan line climbs through the layers;
   where the weight alpha * l/(L-1) * H is high, a retrieval beam rises from the observation memory (the strip of
   visual tokens at the bottom) into that element. Decorative only (aria-hidden); pauses off-screen and for
   reduced motion. */
(function () {
  "use strict";
  SITE.inits.push(function () {
    var cv = document.getElementById("hero-canvas");
    if (!cv || !cv.getContext) return;
    var ctx = cv.getContext("2d");
    var C = SITE.charts;
    var E = SITE.FD && SITE.FD.entropy;
    var NL, NE, Hm;
    if (E && E.H && E.H.length) {
      Hm = E.H; NL = Hm.length; NE = Hm[0].length;
    } else {                                   /* same shape as Figure 1b until the data file is present */
      NL = 32; NE = 56; Hm = [];
      for (var l = 0; l < NL; l++) {
        var row = [];
        for (var n = 0; n < NE; n++) {
          var step = Math.floor(n / 7), settle = 11 + step * 1.8;
          row.push(0.18 + 0.66 / (1 + Math.exp((l - settle) / 1.6)) + 0.04 * Math.sin(n * 1.7 + l));
        }
        Hm.push(row);
      }
    }
    var lo = Infinity, hi = -Infinity;
    Hm.forEach(function (r) { r.forEach(function (v) { lo = Math.min(lo, v); hi = Math.max(hi, v); }); });
    var norm = function (v) { return (v - lo) / (hi - lo || 1); };
    var VIR = C.VIRIDIS.map(function (h) { var p = parseInt(h.slice(1), 16); return [(p >> 16) & 255, (p >> 8) & 255, p & 255]; });
    function vir(t) {
      t = Math.max(0, Math.min(1, t));
      var k = t * (VIR.length - 1), i = Math.min(VIR.length - 2, Math.floor(k)), f = k - i;
      return [VIR[i][0] + (VIR[i + 1][0] - VIR[i][0]) * f, VIR[i][1] + (VIR[i + 1][1] - VIR[i][1]) * f, VIR[i][2] + (VIR[i + 1][2] - VIR[i][2]) * f];
    }

    var W = 0, Ht = 0, dpr = 1, running = false, visible = true, raf = null, last = 0;
    var PASS = 6.5;                            /* seconds per forward pass */
    var phase = 0.35, beams = [], flashes = {};
    var geo = null;
    var NMEM = 28;

    function layout() {
      var mobile = W < 700;
      var x0 = W * (mobile ? 0.04 : 0.06), x1 = W * (mobile ? 0.96 : 0.94);
      var yTop = Ht * 0.16, yBot = Ht * (mobile ? 0.8 : 0.78);
      var cols = mobile ? Math.ceil(NE / 2) : NE;
      geo = {
        x0: x0, x1: x1, yTop: yTop, yBot: yBot, cols: cols, stepX: (x1 - x0) / cols, stepY: (yBot - yTop) / (NL - 1),
        memY: Ht * (mobile ? 0.9 : 0.9), mobile: mobile
      };
      geo.cell = Math.max(2, Math.min(geo.stepX, geo.stepY) * 0.44);
    }
    function colIndex(c) { return geo.mobile ? c * 2 : c; }
    function cellXY(l, c) { return [geo.x0 + geo.stepX * (c + 0.5), geo.yBot - geo.stepY * l]; }
    function memXY(k) { var span = (geo.x1 - geo.x0) * 0.8, x0 = geo.x0 + (geo.x1 - geo.x0) * 0.1; return [x0 + span * (k + 0.5) / NMEM, geo.memY]; }
    /* legibility mask: the title block in the centre stays calm */
    function calm(x, y) {
      var dx = (x - W / 2) / (W * (geo.mobile ? 0.7 : 0.4)), dy = (y - Ht * 0.46) / (Ht * 0.36);
      var d = Math.sqrt(dx * dx + dy * dy);
      return Math.max(0.05, Math.min(1, (d - 0.62) / 0.5));
    }

    function resize() {
      var r = cv.getBoundingClientRect();
      dpr = Math.min(2, window.devicePixelRatio || 1);
      W = Math.max(1, r.width); Ht = Math.max(1, r.height);
      cv.width = Math.round(W * dpr); cv.height = Math.round(Ht * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      layout();
      draw();
    }

    function spawnBeams(l) {
      var depth = l / (NL - 1);
      for (var c = 0; c < geo.cols; c++) {
        var n = colIndex(c), w = depth * norm(Hm[l][n]);
        if (w > 0.42 && Math.random() < 0.18 + 0.5 * (w - 0.42)) {
          beams.push({ l: l, c: c, k: Math.floor(Math.random() * NMEM), t: 0, dur: 0.55 + Math.random() * 0.35, w: w });
        }
      }
      if (beams.length > 90) beams.splice(0, beams.length - 90);
    }

    var lastRow = -1;
    function step(dt) {
      phase += dt / PASS;
      if (phase >= 1.12) { phase = 0; lastRow = -1; flashes = {}; }
      var row = Math.floor(Math.min(1, phase) * (NL - 1));
      if (row !== lastRow && phase <= 1) { for (var r = lastRow + 1; r <= row; r++) if (r > 0) spawnBeams(r); lastRow = row; }
      beams.forEach(function (b) { b.t += dt / b.dur; if (b.t >= 1 && !b.done) { b.done = true; flashes[b.l + ":" + b.c] = 1; } });
      beams = beams.filter(function (b) { return b.t < 1.4; });
      Object.keys(flashes).forEach(function (k) { flashes[k] -= dt * 1.1; if (flashes[k] <= 0) delete flashes[k]; });
    }

    function draw() {
      if (!W || !geo) return;
      ctx.clearRect(0, 0, W, Ht);
      var scan = Math.min(1, phase) * (NL - 1);
      var s = geo.cell;
      /* layer rows */
      for (var l = 0; l < NL; l++) {
        var passed = l <= scan, near = Math.max(0, 1 - Math.abs(l - scan) / 2.2);
        for (var c = 0; c < geo.cols; c++) {
          var n = colIndex(c), p = cellXY(l, c), m = calm(p[0], p[1]);
          var v = norm(Hm[l][n]), rgb = vir(v);
          var a = passed ? (0.14 + 0.5 * v) : 0.05;
          a = (a + near * 0.3) * m;
          var fl = flashes[l + ":" + c];
          if (fl) {
            ctx.fillStyle = "rgba(199,155,240," + (0.55 * fl * m).toFixed(3) + ")";
            ctx.beginPath(); ctx.arc(p[0], p[1], s * (1.4 + 1.6 * (1 - fl)), 0, 6.2832); ctx.fill();
          }
          ctx.fillStyle = "rgba(" + (rgb[0] | 0) + "," + (rgb[1] | 0) + "," + (rgb[2] | 0) + "," + a.toFixed(3) + ")";
          var r = s * (passed ? 0.5 + 0.5 * v : 0.45);
          ctx.fillRect(p[0] - r / 2, p[1] - r / 2, r, r);
        }
      }
      /* scan line */
      var ys = geo.yBot - geo.stepY * scan;
      if (phase <= 1) {
        var g = ctx.createLinearGradient(geo.x0, 0, geo.x1, 0);
        g.addColorStop(0, "rgba(199,155,240,0)");
        g.addColorStop(0.5, "rgba(199,155,240,0.35)");
        g.addColorStop(1, "rgba(199,155,240,0)");
        ctx.strokeStyle = g; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(geo.x0, ys + 0.5); ctx.lineTo(geo.x1, ys + 0.5); ctx.stroke();
      }
      /* observation memory strip */
      for (var k = 0; k < NMEM; k++) {
        var q = memXY(k), mq = calm(q[0], q[1]);
        ctx.fillStyle = "rgba(252,228,152," + (0.22 * Math.max(0.3, mq)).toFixed(3) + ")";
        ctx.fillRect(q[0] - 3.5, q[1] - 3.5, 7, 7);
      }
      /* retrieval beams */
      beams.forEach(function (b) {
        var p0 = memXY(b.k), p1 = cellXY(b.l, b.c);
        var cx = (p0[0] + p1[0]) / 2 + (p1[0] - p0[0]) * 0.15, cy = Math.min(p0[1], p1[1]) + (p0[1] - p1[1]) * 0.25;
        var t = Math.min(1, b.t), fade = b.t > 1 ? Math.max(0, 1 - (b.t - 1) / 0.4) : 1;
        var m = Math.min(calm(p1[0], p1[1]), 1);
        ctx.strokeStyle = "rgba(126,224,176," + (0.16 * fade * m * (0.6 + b.w)).toFixed(3) + ")";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p0[0], p0[1]); ctx.quadraticCurveTo(cx, cy, p1[0], p1[1]); ctx.stroke();
        if (b.t < 1) {
          var u = t, x = (1 - u) * (1 - u) * p0[0] + 2 * (1 - u) * u * cx + u * u * p1[0], y = (1 - u) * (1 - u) * p0[1] + 2 * (1 - u) * u * cy + u * u * p1[1];
          var mm = calm(x, y);
          ctx.fillStyle = "rgba(190,255,220," + (0.85 * mm).toFixed(3) + ")";
          ctx.beginPath(); ctx.arc(x, y, 1.9, 0, 6.2832); ctx.fill();
          ctx.fillStyle = "rgba(126,224,176," + (0.18 * mm).toFixed(3) + ")";
          ctx.beginPath(); ctx.arc(x, y, 6, 0, 6.2832); ctx.fill();
        }
      });
    }

    function frame(ts) {
      if (!running) return;
      if (!last) last = ts;
      var dt = Math.min(0.05, (ts - last) / 1000); last = ts;
      step(dt); draw();
      raf = requestAnimationFrame(frame);
    }
    function start() { if (running || SITE.reduced || !visible || document.hidden) return; running = true; last = 0; raf = requestAnimationFrame(frame); }
    function stop() { running = false; if (raf) cancelAnimationFrame(raf); raf = null; }

    resize();
    if (SITE.reduced) { phase = 1; draw(); }
    window.addEventListener("resize", resize);
    if ("IntersectionObserver" in window) {
      new IntersectionObserver(function (ents) { visible = ents[0].isIntersecting; if (visible) start(); else stop(); }).observe(cv);
    }
    document.addEventListener("visibilitychange", function () { if (document.hidden) stop(); else start(); });
    start();
  });
})();
