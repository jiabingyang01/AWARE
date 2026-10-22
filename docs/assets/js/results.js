/* Section 04: benchmark strip, LIBERO (Table I), SIMPLER (Table II), CALVIN (Table III), real-world setup.
   All numbers come from PAPER_DATA. */
(function () {
  "use strict";
  var D = SITE.D, T = SITE.T, C = SITE.charts, col = SITE.colors;
  var f = SITE.fmt;

  function chip(v) {
    var cls = v > 0 ? "pos" : v < 0 ? "neg" : "zero";
    return '<span class="chip ' + cls + '">' + SITE.signed(v) + "</span>";
  }
  function dot(c) { return '<i style="background:' + c + '"></i>'; }

  /* ---------------------------------------------------------- benchmark strip */
  function bench() {
    var root = document.getElementById("bench-strip"); if (!root) return;
    var S = D.setup;
    [
      { img: "bench_libero.webp", full: "bench_libero_full.webp", t: "LIBERO", d: S.LIBERO.suites.length + " suites × " + S.LIBERO.tasks_per_suite + " tasks · " + S.LIBERO.episodes_per_suite + " episodes per suite", cap: "LIBERO scenes of the four suites: Spatial, Object, Goal and Long." },
      { img: "bench_simpler.webp", full: "bench_simpler_full.webp", t: "SIMPLER", d: S.SIMPLER.protocol + " · Google Robot and WidowX", cap: "SIMPLER Visual Matching scenes: Google Robot (CogACT) and WidowX (SpatialVLA)." },
      { img: "bench_calvin.webp", full: "bench_calvin_full.webp", t: "CALVIN", d: S.CALVIN.split + " · " + S.CALVIN.chains + " chains of five subtasks", cap: "CALVIN environment D, static camera." },
      { img: "bench_real.webp", full: "real_world_setup.webp", t: "Real Franka arm", d: S.real_world.tasks.length + " tabletop tasks · overhead ZED 2i", cap: T.figures.real_world.caption }
    ].forEach(function (it) {
      var b = SITE.h("button", { class: "bench", type: "button", "aria-label": "Open image: " + it.t },
        '<span class="ph"><img src="assets/img/' + it.img + '" width="640" height="400" loading="lazy" decoding="async" alt=""></span>' +
        '<span class="cap"><b>' + it.t + "</b><span>" + SITE.esc(it.d) + "</span></span>");
      b.addEventListener("click", function () { SITE.openLightbox("assets/img/" + it.full, it.t, it.cap, it.t); });
      root.appendChild(b);
    });
  }

  /* ---------------------------------------------------------- LIBERO (Table I) */
  function libero() {
    var L = D.libero; if (!L) return;
    var models = L.models, cols = L.columns, suites = cols.slice(0, 4);
    document.getElementById("lib-caption").innerHTML = "<b>" + L.label + ".</b> " + SITE.pretty(L.caption) +
      " Strengths α for Spatial, Object, Goal and Long: " + models.map(function (m) { return SITE.modelHTML(m.name) + " (" + suites.map(function (s) { return m.alpha[s].toFixed(2); }).join(", ") + ")"; }).join(", ") + ".";

    /* table */
    var h = '<thead><tr><th class="l" scope="col">Method</th>' + cols.map(function (c) { return '<th scope="col">' + c + "</th>"; }).join("") + "</tr></thead><tbody>";
    models.forEach(function (m) {
      h += '<tr><td class="rule l">' + dot(col.Base) + "<b>" + SITE.modelHTML(m.name) + '</b> <span class="dag">' + m.venue + "</span></td>" +
        m.base.map(function (v, i) { return '<td class="' + (v > m.aware[i] ? "best" : "") + '"><span class="v">' + f(v) + "</span></td>"; }).join("") + "</tr>";
      h += '<tr class="is-acc"><td class="rule l">' + dot(col.AWARE) + "w/ AWARE</td>" +
        m.aware.map(function (v, i) { return '<td class="' + (v > m.base[i] ? "best" : "") + '" title="' + (i < 4 ? "α = " + m.alpha[cols[i]].toFixed(2) : "") + '"><span class="v">' + f(v) + "</span></td>"; }).join("") + "</tr>";
      h += '<tr class="delta grp-end"><td class="rule l">Δ</td>' + m.delta.map(function (v) { return "<td>" + chip(v) + "</td>"; }).join("") + "</tr>";
    });
    document.getElementById("lib-table").innerHTML = h + "</tbody>";

    var at = document.getElementById("lib-alpha");
    if (at) at.innerHTML = '<thead><tr><th class="l" scope="col">Backbone</th>' + suites.map(function (s) { return '<th scope="col">' + s + "</th>"; }).join("") + "</tr></thead><tbody>" +
      models.map(function (m) { return '<tr><td class="l">' + SITE.modelHTML(m.name) + "</td>" + suites.map(function (s) { return "<td>" + m.alpha[s].toFixed(2) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody>";

    /* dumbbells: one panel per suite, one row per backbone */
    C.legend(document.getElementById("lib-legend"), [
      { label: "Base policy", color: col.Base, kind: "dot" },
      { label: "w/ AWARE", color: col.AWARE, kind: "dot" }
    ]);
    var el = document.getElementById("lib-dumbbell");
    C.responsive(el, function (w) {
      var narrow = w < 520, rowH = 22, gapG = 16, labW = narrow ? 84 : 104, m = { t: 8, r: 46, b: 40 };
      var groups = cols, h = m.t + groups.length * (models.length * rowH + gapG) - gapG + m.b;
      var svg = C.svg(el, w, h, "Base and w/ AWARE success per LIBERO suite and backbone");
      var x = C.linear(45, 100, labW, w - m.r), g = C.el("g", null, svg);
      C.xAxis(g, x, h - m.b, [50, 60, 70, 80, 90, 100], { grid: true, gridTop: m.t, title: "success (%)" });
      var yy = m.t;
      groups.forEach(function (s, gi) {
        var y0 = yy;
        C.text(g, 0, y0 + (models.length * rowH) / 2 + 4, s, { class: "lbl-strong" });
        models.forEach(function (mm, i) {
          var cy = y0 + i * rowH + rowH / 2, b = mm.base[gi], a = mm.aware[gi];
          var gr = C.el("g", null, g);
          C.el("line", { x1: x(b), x2: x(a), y1: cy, y2: cy, stroke: col.awareLight, "stroke-width": 3, "stroke-linecap": "round", opacity: 0.55 }, gr);
          C.marker(gr, "circle", x(b), cy, 4.6, { fill: col.Base, stroke: "#fff", "stroke-width": 1.5 });
          C.marker(gr, "circle", x(a), cy, 5.2, { fill: col.AWARE, stroke: "#fff", "stroke-width": 1.5 });
          var lab = C.el("text", { x: labW - 8 - (narrow ? 0 : 0), y: cy + 4, class: "lbl-mono", "text-anchor": "end" }, gr);
          lab.innerHTML = C.modelSVG(narrow ? mm.name.replace("OpenVLA-OFT", "OFT") : mm.name);
          C.text(gr, w - 2, cy + 4, SITE.signed(mm.delta[gi]), { class: "lbl-mono", "text-anchor": "end", style: "fill:" + (mm.delta[gi] > 0 ? col.pos : col.neg) + ";font-weight:650" });
          var hit = C.el("rect", { x: labW - 90, y: cy - rowH / 2, width: w - labW + 90, height: rowH, class: "hit" }, gr);
          C.hover(hit, function () {
            return '<div class="th">' + SITE.modelHTML(mm.name) + " · " + (s === "Average" ? "average of the four suites" : "LIBERO-" + s) + "</div>" +
              C.row("Base", f(b), col.Base) + C.row("w/ AWARE", f(a), col.AWARE) + C.row("Δ", SITE.signed(mm.delta[gi])) +
              (gi < 4 ? '<div class="tn">α = ' + mm.alpha[s].toFixed(2) + "</div>" : "");
          });
        });
        if (gi < groups.length - 1) C.el("line", { x1: 0, x2: w, y1: y0 + models.length * rowH + gapG / 2, y2: y0 + models.length * rowH + gapG / 2, stroke: "rgba(22,20,29,.08)" }, g);
        yy += models.length * rowH + gapG;
      });
    });

    /* grouped gain bars */
    var el2 = document.getElementById("lib-gain");
    C.responsive(el2, function (w) {
      var h = 300, m = { l: 36, r: 8, t: 26, b: 44 };
      var svg = C.svg(el2, w, h, "Gain of AWARE per suite and backbone");
      var maxD = Math.max.apply(null, models.map(function (mm) { return Math.max.apply(null, mm.delta); }));
      var y = C.linear(0, Math.ceil(maxD + 0.4), h - m.b, m.t), band = C.band(cols.length, m.l, w - m.r, 0.18);
      var g = C.el("g", null, svg);
      C.yAxis(g, y, m.l, w - m.r, C.ticks(0, Math.ceil(maxD + 0.4), 4), { decimals: 0 });
      var bw = band.inner / models.length;
      cols.forEach(function (s, gi) {
        var cx = band(gi);
        if (s === "Long") C.el("rect", { x: cx - band.bw / 2 + 2, y: m.t - 18, width: band.bw - 4, height: h - m.b - m.t + 18, rx: 8, fill: "rgba(112,48,160,.06)" }, g);
        models.forEach(function (mm, i) {
          var v = mm.delta[gi], x0 = cx - band.inner / 2 + i * bw;
          var r = C.el("rect", { x: x0 + 1, y: y(v), width: Math.max(2, bw - 2), height: y(0) - y(v), rx: 2, fill: col.backbones[mm.name] }, g);
          C.hover(r, function () { return '<div class="th">' + SITE.modelHTML(mm.name) + " · " + s + "</div>" + C.row("Δ", SITE.signed(v), col.backbones[mm.name]) + C.row("Base → AWARE", f(mm.base[gi]) + " → " + f(mm.aware[gi])); });
        });
        C.text(g, cx, h - m.b + 18, s, { "text-anchor": "middle", class: s === "Long" ? "lbl-strong" : "lbl" });
      });
      C.text(g, 4, m.t - 12, "Δ (points)", { class: "lbl-mono" });
      var lx = m.l + 4;
      models.forEach(function (mm) {
        var t = C.el("text", { x: 0, y: h - 8, class: "lbl-mono" }, g);
        t.innerHTML = C.modelSVG(mm.name);
        var tw = 10 + mm.name.length * 6.2;
        C.el("rect", { x: lx, y: h - 16, width: 9, height: 9, rx: 2, fill: col.backbones[mm.name] }, g);
        t.setAttribute("x", lx + 13); lx += tw + 22;
      });
    });
  }

  /* ---------------------------------------------------------- SIMPLER (Table II) */
  function simpler() {
    var S = D.simpler; if (!S) return;
    document.getElementById("sim-caption").innerHTML = "<b>" + S.label + ".</b> " + SITE.pretty(S.caption) + " Strengths: α = " + S.blocks[0].alpha.toFixed(2) + " for CogACT and α = " + S.blocks[1].alpha.toFixed(2) + " for SpatialVLA.";
    var h = "";
    S.blocks.forEach(function (B, bi) {
      h += (bi ? "" : "<thead>") + '<tr class="' + (bi ? "grp-hd sub-hd" : "") + '"><th class="l" scope="col">' + B.robot + "</th>" + B.columns.map(function (c) { return '<th scope="col" title="' + c + '">' + c.replace("Open Top Drawer and Place Apple", "Drawer + Apple").replace("Stack Green Block on Yellow Block", "Stack Block").replace("Put Eggplant in Yellow Basket", "Eggplant in Basket") + "</th>"; }).join("") + "</tr>" + (bi ? "" : "</thead><tbody>");
      h += '<tr><td class="rule l">' + dot(col.Base) + "<b>" + B.model + '</b> <span class="dag">' + B.venue + "</span></td>" + B.base.map(function (v, i) { return '<td class="' + (v > B.aware[i] ? "best" : "") + '"><span class="v">' + f(v) + "</span></td>"; }).join("") + "</tr>";
      h += '<tr class="is-acc"><td class="rule l">' + dot(col.AWARE) + "w/ AWARE</td>" + B.aware.map(function (v, i) { return '<td class="' + (v > B.base[i] ? "best" : "") + '"><span class="v">' + f(v) + "</span></td>"; }).join("") + "</tr>";
      h += '<tr class="delta grp-end"><td class="rule l">Δ</td>' + B.delta.map(function (v) { return "<td>" + chip(v) + "</td>"; }).join("") + "</tr>";
    });
    document.getElementById("sim-table").innerHTML = h + "</tbody>";
    document.getElementById("sim-g-sub").textContent = "α = " + S.blocks[0].alpha.toFixed(2) + "; the largest gain is on the multi-stage drawer-and-apple task.";
    document.getElementById("sim-w-sub").textContent = "α = " + S.blocks[1].alpha.toFixed(2) + "; " + S.widowx_pairs.evaluations + " paired evaluations, " + S.widowx_pairs.wins + " wins and " + S.widowx_pairs.losses + " losses.";
    [["sim-g-chart", S.blocks[0]], ["sim-w-chart", S.blocks[1]]].forEach(function (p) {
      var el = document.getElementById(p[0]), B = p[1];
      C.responsive(el, function (w) {
        var rowH = 38, labW = Math.min(170, w * 0.42), m = { t: 6, r: 44, b: 38 }, h = m.t + B.columns.length * rowH + m.b;
        var svg = C.svg(el, w, h, B.robot + " success, base and with AWARE");
        var x = C.linear(0, 100, labW, w - m.r), g = C.el("g", null, svg);
        C.xAxis(g, x, h - m.b, [0, 25, 50, 75, 100], { grid: true, gridTop: m.t, title: "success (%)" });
        B.columns.forEach(function (c, i) {
          var cy = m.t + i * rowH + rowH / 2, b = B.base[i], a = B.aware[i], avg = i === B.columns.length - 1;
          var gr = C.el("g", null, g);
          var words = c.split(" "), lines = [], cur = "";
          words.forEach(function (wd) { if ((cur + " " + wd).trim().length > 22) { lines.push(cur.trim()); cur = wd; } else cur += " " + wd; });
          lines.push(cur.trim());
          lines.forEach(function (ln, k) { C.text(gr, labW - 10, cy + 4 + (k - (lines.length - 1) / 2) * 13, ln, { class: avg ? "lbl-strong" : "lbl", "text-anchor": "end" }); });
          C.el("line", { x1: x(b), x2: x(a), y1: cy, y2: cy, stroke: a >= b ? col.awareLight : "#e8a0ad", "stroke-width": 3, "stroke-linecap": "round", opacity: 0.7 }, gr);
          C.marker(gr, "circle", x(b), cy, 4.8, { fill: col.Base, stroke: "#fff", "stroke-width": 1.5 });
          C.marker(gr, "circle", x(a), cy, 5.4, { fill: col.AWARE, stroke: "#fff", "stroke-width": 1.5 });
          C.text(gr, w - 2, cy + 4, SITE.signed(B.delta[i]), { class: "lbl-mono", "text-anchor": "end", style: "fill:" + (B.delta[i] > 0 ? col.pos : col.neg) + ";font-weight:650" });
          var hit = C.el("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, class: "hit" }, gr);
          C.hover(hit, function () { return '<div class="th">' + B.model + " · " + c + "</div>" + C.row("Base", f(b), col.Base) + C.row("w/ AWARE", f(a), col.AWARE) + C.row("Δ", SITE.signed(B.delta[i])); });
        });
      });
    });
  }

  /* ---------------------------------------------------------- CALVIN (Table III) */
  function calvin() {
    var K = D.calvin; if (!K) return;
    document.getElementById("cal-caption").innerHTML = "<b>" + K.label + ".</b> " + SITE.pretty(K.caption);
    var cols = K.columns.concat(["Avg. Len"]);
    var h = '<thead><tr><th class="l" scope="col">Method</th>' + cols.map(function (c) { return '<th scope="col">' + c + "</th>"; }).join("") + "</tr></thead><tbody>";
    h += '<tr><td class="rule l">' + dot(col.Base) + "<b>" + K.model + '</b> <span class="dag">' + K.venue + "</span></td>" + K.base.map(function (v) { return '<td><span class="v">' + f(v) + "</span></td>"; }).join("") + '<td><span class="v">' + f(K.avg_len.base, 2) + "</span></td></tr>";
    h += '<tr class="is-acc"><td class="rule l">' + dot(col.AWARE) + "w/ AWARE</td>" + K.aware.map(function (v) { return '<td class="best"><span class="v">' + f(v) + "</span></td>"; }).join("") + '<td class="best"><span class="v">' + f(K.avg_len.aware, 2) + "</span></td></tr>";
    h += '<tr class="delta"><td class="rule l">Δ</td>' + K.delta.map(function (v) { return "<td>" + chip(v) + "</td>"; }).join("") + '<td><span class="chip pos">+' + f(K.avg_len.delta, 2) + "</span></td></tr>";
    document.getElementById("cal-table").innerHTML = h + "</tbody>";
    document.getElementById("cal-stats").innerHTML =
      '<div class="stat"><div class="k">Average completed length</div><div class="v">' + f(K.avg_len.base, 2) + ' <small>→</small> <span style="color:' + col.awareDeep + '">' + f(K.avg_len.aware, 2) + '</span></div><div class="d">subtasks per chain of five</div></div>' +
      '<div class="stat"><div class="k">All five subtasks</div><div class="v">' + f(K.base[4]) + ' <small>→</small> <span style="color:' + col.awareDeep + '">' + f(K.aware[4]) + '</span><small>%</small></div><div class="d">complete chains</div></div>';
    C.legend(document.getElementById("cal-legend"), [{ label: "LLaVA-VLA (base)", color: col.Base, kind: "line" }, { label: "w/ AWARE", color: col.AWARE, kind: "line" }, { label: "gain", color: col.AWARE, kind: "band" }]);
    var el = document.getElementById("cal-chart");
    C.responsive(el, function (w) {
      var h = 300, m = { l: 44, r: 16, t: 14, b: 44 };
      var svg = C.svg(el, w, h, "CALVIN success by chain length");
      var x = C.linear(0.7, 5.3, m.l, w - m.r), y = C.linear(40, 100, h - m.b, m.t), g = C.el("g", null, svg);
      C.yAxis(g, y, m.l, w - m.r, [40, 50, 60, 70, 80, 90, 100], { title: "success (%)", titleX: 10 });
      C.xAxis(g, x, h - m.b, [1, 2, 3, 4, 5], { format: function (t) { return t + "/5"; }, title: "completed subtasks in the chain" });
      var band = K.base.map(function (v, i) { return [x(i + 1), y(K.aware[i])]; }).concat(K.base.map(function (v, i) { return [x(i + 1), y(v)]; }).reverse());
      C.el("path", { d: "M" + band.map(function (p) { return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join("L") + "Z", fill: col.AWARE, opacity: 0.12 }, g);
      [[K.base, col.Base], [K.aware, col.AWARE]].forEach(function (p) {
        C.el("path", { d: C.path(p[0].map(function (v, i) { return [x(i + 1), y(v)]; })), fill: "none", stroke: p[1], "stroke-width": 2.4 }, g);
        p[0].forEach(function (v, i) { C.marker(g, "circle", x(i + 1), y(v), 4.6, { fill: p[1], stroke: "#fff", "stroke-width": 1.5 }); });
      });
      K.delta.forEach(function (d, i) {
        C.text(g, x(i + 1), y(K.aware[i]) - 11, SITE.signed(d), { "text-anchor": "middle", class: "lbl-mono", style: "fill:" + col.awareDeep + ";font-weight:650" });
        var hit = C.el("rect", { x: x(i + 1) - 24, y: m.t, width: 48, height: h - m.b - m.t, class: "hit" }, g);
        C.hover(hit, function () { return '<div class="th">' + (i + 1) + " of 5 subtasks completed</div>" + C.row("LLaVA-VLA", f(K.base[i]), col.Base) + C.row("w/ AWARE", f(K.aware[i]), col.AWARE) + C.row("Δ", SITE.signed(d)); });
      });
    });
  }

  /* ---------------------------------------------------------- real world: setup only */
  function real() {
    var root = document.getElementById("real-root"); if (!root) return;
    var R = D.setup.real_world, F = T.figures.real_world;
    root.innerHTML =
      '<figure class="card figure-card"><img src="' + F.files.webp + '" loading="lazy" decoding="async" data-zoom alt="' + SITE.esc(F.alt) + '" data-caption="' + SITE.esc(F.caption) + '" data-label="' + F.group + '">' +
      '<figcaption class="caption"><b>' + F.group + ".</b> " + SITE.pretty(F.caption) + "</figcaption></figure>" +
      '<div class="facts"><span class="fact"><b>' + SITE.esc(R.robot) + '</b></span><span class="fact">' + SITE.esc(R.camera) + ' camera</span><span class="fact"><b>' + R.demos_per_task + "</b> teleoperated demonstrations per task · " + SITE.esc(R.teleoperation) + "</span>" +
      "</div><div class=\"facts\"><span class=\"fact-lab\">Strength α per backbone</span>" + R.backbones.map(function (b) { return '<span class="fact">' + SITE.modelHTML(b) + ": α&nbsp;=&nbsp;" + R.alpha[b].toFixed(2) + "</span>"; }).join("") + "</div>" +
      '<ol class="rw-tasks">' + R.tasks.map(function (t, i) {
        return '<li class="rw-task"><b>' + (i + 1) + ". " + SITE.esc(t.name) + "</b><span>“" + SITE.esc(t.instruction) + "”</span>" + (t.moving ? "<em>moving target</em>" : '<em style="color:var(--muted);background:rgba(22,20,29,.05)">' + SITE.esc(t.kind) + "</em>") + "</li>";
      }).join("") + "</ol>";
    SITE.$$("img[data-zoom]", root).forEach(SITE.zoomable);
  }

  function scope() {
    var ul = document.getElementById("scope-list"); if (!ul || !T.scope) return;
    ul.innerHTML = T.scope.map(function (s) { return "<li>" + SITE.pretty(s) + "</li>"; }).join("");
  }

  SITE.inits.push(bench, libero, simpler, calvin, real, scope);
})();
