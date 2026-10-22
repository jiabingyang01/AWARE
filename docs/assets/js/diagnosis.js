/* Section 02: Figure 1 redrawn from its data (FIG_DATA.footprint and FIG_DATA.entropy). */
(function () {
  "use strict";
  var FD = SITE.FD, C = SITE.charts, col = SITE.colors;
  var SUITES = ["Spatial", "Object", "Goal", "Long"];
  var DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"];

  /* ---------------------------------------------------------- (a) observation share */
  function footprint() {
    var F = FD.footprint, el = document.getElementById("fp-chart");
    if (!F || !el) return;
    C.legend(document.getElementById("fp-legend"), SUITES.map(function (s) { return { label: s, color: F.colors[s], kind: "line" }; }));
    C.responsive(el, function (w) {
      var h = Math.max(240, Math.min(330, w * 0.62)), m = { l: 46, r: 14, t: 14, b: 44 };
      var svg = C.svg(el, w, h, "Observation share of the action representation by layer");
      var L = F.layers, x = C.linear(-0.7, L.length - 0.3, m.l, w - m.r), y = C.linear(F.ylim[0], F.ylim[1], h - m.b, m.t);
      var g = C.el("g", null, svg);
      C.el("rect", { x: x(F.band[0] - 0.5), y: m.t, width: x(F.band[1] + 0.5) - x(F.band[0] - 0.5), height: h - m.b - m.t, fill: "rgba(22,20,29,.045)" }, g);
      C.yAxis(g, y, m.l, w - m.r, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], { title: "observation share", titleX: 10 });
      C.xAxis(g, x, h - m.b, [0, 8, 16, 24, 31], { title: "layer ℓ" });
      var xh = x(26);
      C.el("line", { x1: xh, x2: xh, y1: m.t, y2: h - m.b, stroke: col.AWARE, "stroke-dasharray": "3 3", "stroke-width": 1.2, opacity: 0.7 }, g);
      C.text(g, xh - 6, m.t + 12, "≈ ½ at layer 26", { class: "lbl-mono", "text-anchor": "end", style: "fill:" + col.awareDeep + ";font-weight:600" });
      SUITES.forEach(function (s) {
        var ys = F.suites[s];
        C.el("path", { d: C.path(ys.map(function (v, i) { return [x(L[i]), y(v)]; })), fill: "none", stroke: F.colors[s], "stroke-width": 2, "stroke-linejoin": "round" }, g);
        [0, 6, 12, 18, 24, 30].forEach(function (i) {
          C.marker(g, col.suiteShapes[s], x(L[i]), y(ys[i]), 3.4, { fill: F.colors[s], stroke: "#fff", "stroke-width": 1.2 });
        });
      });
      var hl = C.el("line", { class: "hover-line", y1: m.t, y2: h - m.b }, g);
      var dots = SUITES.map(function (s) { return C.el("circle", { r: 4.5, fill: F.colors[s], stroke: "#fff", "stroke-width": 1.5, opacity: 0 }, g); });
      var hit = C.el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.b - m.t, class: "hit" }, svg);
      var li = 26;
      function at(evt) {
        if (evt && evt.clientX != null) {
          var r = svg.getBoundingClientRect();
          li = Math.max(0, Math.min(L.length - 1, Math.round(x.invert(evt.clientX - r.left))));
        }
        hl.setAttribute("x1", x(li)); hl.setAttribute("x2", x(li)); hl.classList.add("on");
        dots.forEach(function (d, k) { d.setAttribute("cx", x(li)); d.setAttribute("cy", y(F.suites[SUITES[k]][li])); d.setAttribute("opacity", 1); });
        return '<div class="th">Layer ' + li + "</div>" + SUITES.map(function (s) { return C.row(s, SITE.fmt(F.suites[s][li], 3), F.colors[s]); }).join("");
      }
      hit.addEventListener("pointermove", function (e) { C.tip.show(at(e), e, hit); });
      C.hover(hit, function () { return at(null); }, null, function () { hl.classList.remove("on"); dots.forEach(function (d) { d.setAttribute("opacity", 0); }); });
    });
  }

  /* ---------------------------------------------------------- (b) Action Entropy heatmap */
  function entropy() {
    var E = FD.entropy, el = document.getElementById("ent-chart");
    if (!E || !el) return;
    var state = { mode: "elements" };
    SITE.segmented(document.getElementById("ent-mode"), [
      { value: "elements", label: "56 elements" },
      { value: "steps", label: "8 steps <small>(mean over dims)</small>" }
    ], state.mode, function (v) { state.mode = v; resp.redraw(); });
    var sc = document.getElementById("ent-scale");
    sc.innerHTML = '<div class="bar" style="background:linear-gradient(90deg,' + C.VIRIDIS.join(",") + ')"></div><div class="ticks"><span>' + E.vmin + '</span><span>Action Entropy</span><span>' + E.vmax + "</span></div>";
    var norm = function (v) { return (v - E.vmin) / (E.vmax - E.vmin); };
    var NL = E.H.length;
    var resp = C.responsive(el, function (w) {
      var steps = state.mode === "steps", NC = steps ? E.steps : E.H[0].length;
      var m = { l: 40, r: 8, t: 8, b: 40 }, h = Math.max(250, Math.min(340, w * 0.66));
      var svg = C.svg(el, w, h, "Action Entropy by layer and chunk element");
      var cw = (w - m.l - m.r) / NC, ch = (h - m.t - m.b) / NL;
      var g = C.el("g", null, svg);
      for (var l = 0; l < NL; l++) {
        for (var c = 0; c < NC; c++) {
          var v = steps ? E.step_means[l][c] : E.H[l][c];
          C.el("rect", { x: m.l + c * cw, y: h - m.b - (l + 1) * ch, width: Math.ceil(cw) + 0.4, height: Math.ceil(ch) + 0.4, fill: C.ramp(C.VIRIDIS, norm(v)) }, g);
        }
      }
      if (!steps) for (var k = 1; k < E.steps; k++) C.el("line", { x1: m.l + k * E.dims * cw, x2: m.l + k * E.dims * cw, y1: m.t, y2: h - m.b, stroke: "rgba(255,255,255,.35)", "stroke-width": 1 }, g);
      /* the paper's dashed wavefront: fitted through the layer where each step crosses the level */
      var X = function (xe) { return steps ? m.l + (xe - (E.dims - 1) / 2) / E.dims * cw + cw / 2 : m.l + (xe + 0.5) * cw; };
      var Y = function (yl) { return h - m.b - (yl + 0.5) * ch; };
      C.el("path", { d: C.path(E.wavefront.x.map(function (xx, i) { return [X(xx), Y(E.wavefront.y[i])]; })), fill: "none", stroke: "#fff", "stroke-width": 2, "stroke-dasharray": "5 3", opacity: 0.95 }, g);
      var ax = C.el("g", { class: "axis" }, g);
      for (var s = 0; s < E.steps; s++) {
        var cx = steps ? m.l + (s + 0.5) * cw : m.l + (s * E.dims + E.dims / 2) * cw;
        C.text(ax, cx, h - m.b + 16, String(s + 1), { "text-anchor": "middle" });
      }
      C.text(ax, (m.l + w - m.r) / 2, h - 6, "future step of the chunk", { class: "axis-title", "text-anchor": "middle" });
      [0, 8, 16, 24, 31].forEach(function (t) { C.text(ax, m.l - 7, Y(t) + 4, String(t), { "text-anchor": "end" }); });
      C.text(ax, 0, 0, "layer ℓ", { class: "axis-title", transform: "translate(11," + ((m.t + h - m.b) / 2) + ") rotate(-90)", "text-anchor": "middle" });
      var hover = C.el("rect", { fill: "none", stroke: "#fff", "stroke-width": 2, width: cw, height: ch, opacity: 0, "pointer-events": "none" }, g);
      var hit = C.el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, class: "hit" }, svg);
      var cur = { l: 20, c: steps ? 7 : 52 };
      function at(evt) {
        if (evt && evt.clientX != null) {
          var r = svg.getBoundingClientRect();
          cur.c = Math.max(0, Math.min(NC - 1, Math.floor((evt.clientX - r.left - m.l) / cw)));
          cur.l = Math.max(0, Math.min(NL - 1, Math.floor((h - m.b - (evt.clientY - r.top)) / ch)));
        }
        hover.setAttribute("x", m.l + cur.c * cw); hover.setAttribute("y", h - m.b - (cur.l + 1) * ch); hover.setAttribute("opacity", 1);
        var v = steps ? E.step_means[cur.l][cur.c] : E.H[cur.l][cur.c];
        var name = steps ? "Step " + (cur.c + 1) + " (mean of 7 dims)" : "Step " + (Math.floor(cur.c / E.dims) + 1) + " · " + DIMS[cur.c % E.dims];
        return '<div class="th">' + name + "</div>" + C.row("layer", String(cur.l)) + C.row("Action Entropy", SITE.fmt(v, 3), C.ramp(C.VIRIDIS, norm(v)));
      }
      hit.addEventListener("pointermove", function (e) { C.tip.show(at(e), e, hit); });
      C.hover(hit, function () { return at(null); }, null, function () { hover.setAttribute("opacity", 0); });
    });
  }

  /* ---------------------------------------------------------- measured facts */
  function callouts() {
    var F = FD.footprint, E = FD.entropy, a = document.getElementById("fp-note"), b = document.getElementById("ent-note");
    if (!F || !E || !a || !b) return;
    var at26 = SUITES.map(function (s) { return s + " " + SITE.fmt(F.at_26[s], 2); }).join(" · ");
    var st = E.wavefront.settle, gap = 0, gl = 0;
    E.step_means.forEach(function (r, l) { var d = r[E.steps - 1] - r[0]; if (d > gap) { gap = d; gl = l; } });
    a.innerHTML = '<span class="tag">At layer 26</span><div class="ct">About <b>half</b> of the initial share, in every suite</div><div class="cn">' + at26 + "</div>";
    b.innerHTML = '<span class="tag">Along the dashed wavefront</span><div class="ct">Step 1 settles near layer <b>' + Math.round(st[0]) + "</b>, step 8 near layer <b>" + Math.round(st[st.length - 1]) + "</b></div>" +
      '<div class="cn">largest step 8 − step 1 entropy gap: ' + SITE.fmt(gap, 2) + " at layer " + gl + "</div>";
  }

  SITE.inits.push(footprint, entropy, callouts);
})();
