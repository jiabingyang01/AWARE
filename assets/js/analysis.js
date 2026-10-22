/* Section 05: component analysis (Table IV), diagnostic variants (Table V), strength sensitivity (Figure 5),
   inference overhead (Table VI). */
(function () {
  "use strict";
  var D = SITE.D, FD = SITE.FD, C = SITE.charts, col = SITE.colors;
  var f = SITE.fmt;

  function ablation() {
    var A = D.ablation, el = document.getElementById("abl-chart"); if (!A || !el) return;
    var rows = A.rows, iAvg = A.columns.length - 1, base = rows[0].values[iAvg];
    C.responsive(el, function (w) {
      var narrow = w < 560, rowH = narrow ? 60 : 46, labW = narrow ? 8 : Math.min(190, w * 0.44), m = { t: 8, r: 48, b: 40 }, h = m.t + rows.length * rowH + m.b;
      var svg = C.svg(el, w, h, "Component analysis, average success");
      var x = C.linear(96, 98.5, labW, w - m.r), g = C.el("g", null, svg);
      C.xAxis(g, x, h - m.b, [96, 96.5, 97, 97.5, 98, 98.5], { grid: true, gridTop: m.t, title: "average success over the four suites (%)" });
      C.el("line", { x1: x(base), x2: x(base), y1: m.t, y2: h - m.b, stroke: col.Base, "stroke-dasharray": "4 3", "stroke-width": 1.2 }, g);
      rows.forEach(function (r, i) {
        var cy = m.t + i * rowH + rowH / 2 + (narrow ? 10 : 0), v = r.values[iAvg], full = r.key === "full", isB = r.key === "base";
        var gr = C.el("g", null, g);
        var parts = [["element", r.per_element], ["depth", r.depth], ["instruction", r.instruction]].map(function (p) { return (p[1] ? "✓ " : "✕ ") + p[0]; }).join("  ");
        if (narrow) {
          C.text(gr, 0, cy - 16, r.name, { class: full ? "lbl-strong" : "lbl", style: full ? "fill:" + col.awareDeep : "" });
        } else {
          C.text(gr, labW - 10, cy - 2, r.name, { class: full ? "lbl-strong" : "lbl", "text-anchor": "end", style: full ? "fill:" + col.awareDeep : "" });
          C.text(gr, labW - 10, cy + 13, parts, { class: "lbl-mono", "text-anchor": "end", style: "font-size:10.5px" });
        }
        if (!isB) C.el("rect", { x: x(base), y: cy - 9, width: Math.max(1, x(v) - x(base)), height: 18, rx: 4, fill: full ? col.AWARE : col.awareLight, opacity: full ? 1 : 0.55 }, gr);
        C.marker(gr, "circle", x(v), cy, isB ? 5 : 4, { fill: isB ? col.Base : full ? col.awareDeep : col.AWARE, stroke: "#fff", "stroke-width": 1.5 });
        C.text(gr, w - 2, cy + 4, f(v), { class: full ? "lbl-strong" : "lbl", "text-anchor": "end" });
        var hit = C.el("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, class: "hit" }, gr);
        C.hover(hit, function () {
          return '<div class="th">' + r.name + "</div>" + A.columns.map(function (c, k) { return C.row(c, f(r.values[k]) + (isB ? "" : "  (" + SITE.signed(r.values[k] - rows[0].values[k]) + ")")); }).join("");
        });
      });
    });
  }

  function controls() {
    var K = D.controls, el = document.getElementById("ctl-chart"); if (!K || !el) return;
    var items = [{ name: "AWARE (full rule)", delta: K.aware_delta, value: K.aware, full: true, aspect: "Reference" }];
    K.groups.forEach(function (g) { g.rows.forEach(function (r) { items.push({ name: r.name, delta: r.delta, value: r.value, aspect: g.aspect }); }); });
    C.responsive(el, function (w) {
      var narrow = w < 560, rowH = narrow ? 46 : 34, labW = narrow ? 8 : Math.min(210, w * 0.5), m = { t: 22, r: 44, b: 40 }, gaps = 2, h = m.t + items.length * rowH + gaps * 14 + m.b + 8;
      var svg = C.svg(el, w, h, "Diagnostic variants, gain over the base policy");
      var x = C.linear(0, 2.5, labW, w - m.r), g = C.el("g", null, svg);
      C.xAxis(g, x, h - m.b, [0, 0.5, 1, 1.5, 2, 2.5], { grid: true, gridTop: m.t, title: "gain over the base policy on LIBERO-Long (points)" });
      C.el("line", { x1: x(K.aware_delta), x2: x(K.aware_delta), y1: m.t - 8, y2: h - m.b, stroke: col.AWARE, "stroke-dasharray": "4 3", "stroke-width": 1.2 }, g);
      C.text(g, x(K.aware_delta), m.t - 12, "AWARE +" + f(K.aware_delta), { "text-anchor": "middle", class: "lbl-mono", style: "fill:" + col.awareDeep + ";font-weight:650" });
      var yy = m.t, lastAspect = null;
      items.forEach(function (it) {
        if (it.aspect !== lastAspect) {
          if (lastAspect) yy += 14;
          if (!it.full && !narrow) C.text(g, 0, yy + 2, it.aspect.toUpperCase(), { class: "lbl-mono", style: "font-size:10.5px;letter-spacing:.06em;font-weight:650" });
        }
        lastAspect = it.aspect;
        var cy = yy + rowH / 2 + (narrow ? 8 : 0), gr = C.el("g", null, g);
        if (narrow) C.text(gr, 0, cy - 13, it.name, { class: it.full ? "lbl-strong" : "lbl", style: it.full ? "fill:" + col.awareDeep : "" });
        else C.text(gr, labW - 10, cy + 4, it.name, { class: it.full ? "lbl-strong" : "lbl", "text-anchor": "end", style: it.full ? "fill:" + col.awareDeep : "" });
        C.el("rect", { x: x(0), y: cy - 8, width: Math.max(1, x(it.delta) - x(0)), height: 16, rx: 4, fill: it.full ? col.AWARE : "#b9b3c6" }, gr);
        C.text(gr, w - 2, cy + 4, "+" + f(it.delta), { class: it.full ? "lbl-strong" : "lbl", "text-anchor": "end" });
        var hit = C.el("rect", { x: 0, y: cy - rowH / 2, width: w, height: rowH, class: "hit" }, gr);
        C.hover(hit, function () { return '<div class="th">' + SITE.esc(it.name) + "</div>" + C.row("tested aspect", it.aspect) + C.row("success", f(it.value)) + C.row("Δ vs. base " + f(K.base), "+" + f(it.delta)); });
        yy += rowH;
      });
    });
  }

  function alpha() {
    var AL = FD.alpha, el = document.getElementById("alpha-chart"); if (!AL || !el) return;
    var SU = ["Spatial", "Object", "Goal", "Long"], names = Object.keys(AL), state = { m: names[0] };
    SITE.segmented(document.getElementById("alpha-model"), names.map(function (n) { return { value: n, label: SITE.modelHTML(n), dot: col.backbones[n] }; }), state.m, function (v) { state.m = v; resp.redraw(); side(); });
    document.getElementById("alpha-sub").innerHTML = "Success per LIBERO suite as α sweeps from 0 (the base policy) to 0.10; outlined markers are the strengths used in Table I. Hover for the values at one α.";
    C.legend(document.getElementById("alpha-legend"), SU.map(function (s) { return { label: s, color: AL[state.m].colors[s], kind: "line" }; }));
    var resp = C.responsive(el, function (w) {
      var P = AL[state.m], h = Math.max(260, Math.min(360, w * 0.5)), m = { l: 44, r: 14, t: 12, b: 44 };
      var lo = Math.floor(Math.min.apply(null, SU.map(function (s) { return Math.min.apply(null, P.suites[s]); })) - 1), hi = 100;
      var svg = C.svg(el, w, h, "Success per suite versus strength alpha");
      var x = C.linear(-0.004, 0.104, m.l, w - m.r), y = C.linear(lo, hi, h - m.b, m.t), g = C.el("g", null, svg);
      C.yAxis(g, y, m.l, w - m.r, C.ticks(lo, hi, 5), { title: "success (%)", titleX: 10, decimals: 0 });
      C.xAxis(g, x, h - m.b, [0, 0.02, 0.04, 0.06, 0.08, 0.1], { format: function (t) { return t === 0 ? "0" : t.toFixed(2); }, title: "strength α" });
      SU.forEach(function (s) {
        var c = P.colors[s], ys = P.suites[s];
        C.el("path", { d: C.path(ys.map(function (v, i) { return [x(P.alphas[i]), y(v)]; })), fill: "none", stroke: c, "stroke-width": 2.2, "stroke-dasharray": "6 3" }, g);
        ys.forEach(function (v, i) {
          var best = Math.abs(P.alphas[i] - P.best[s]) < 1e-9;
          C.marker(g, col.suiteShapes[s], x(P.alphas[i]), y(v), best ? 6 : 4, { fill: c, stroke: best ? "#16141d" : "#fff", "stroke-width": best ? 1.6 : 1.2 });
        });
      });
      var hl = C.el("line", { class: "hover-line", y1: m.t, y2: h - m.b }, g);
      var hit = C.el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, class: "hit" }, svg);
      var ai = 3;
      function at(evt) {
        if (evt && evt.clientX != null) {
          var r = svg.getBoundingClientRect(), a = x.invert(evt.clientX - r.left);
          ai = Math.max(0, Math.min(P.alphas.length - 1, Math.round(a / 0.01)));
        }
        hl.setAttribute("x1", x(P.alphas[ai])); hl.setAttribute("x2", x(P.alphas[ai])); hl.classList.add("on");
        return '<div class="th">' + SITE.modelHTML(state.m) + " · α = " + P.alphas[ai].toFixed(2) + (ai === 0 ? " (base)" : "") + "</div>" +
          SU.map(function (s) { return C.row(s, f(P.suites[s][ai]) + (ai ? "  (" + SITE.signed(P.suites[s][ai] - P.suites[s][0]) + ")" : ""), P.colors[s]); }).join("");
      }
      hit.addEventListener("pointermove", function (e) { C.tip.show(at(e), e, hit); });
      C.hover(hit, function () { return at(null); }, null, function () { hl.classList.remove("on"); });
    });
    function side() {
      var P = AL[state.m], root = document.getElementById("alpha-side");
      root.innerHTML = '<h3 class="label">Strength of Table I, per suite</h3>' + SU.map(function (s) {
        var i = P.alphas.map(function (a) { return Math.round(a * 100); }).indexOf(Math.round(P.best[s] * 100));
        return '<div class="kv"><span>' + s + "</span><b>α = " + P.best[s].toFixed(2) + "<small>" + f(P.suites[s][0]) + " → " + f(P.suites[s][i]) + "</small></b></div>";
      }).join("") + '<p class="note">On LIBERO-Long both backbones peak at α = 0.03, with +' + f(D.text_numbers.alpha_gain_long["OpenVLA-OFT"]) + " for OpenVLA-OFT and +" + f(D.text_numbers.alpha_gain_long["π0"]) + " for " + SITE.modelHTML("π0") + ".</p>";
      C.legend(document.getElementById("alpha-legend"), SU.map(function (s) { return { label: s, color: P.colors[s], kind: "line" }; }));
    }
    side();
  }

  function cost() {
    var O = D.overhead, root = document.getElementById("cost-grid"); if (!O || !root) return;
    root.innerHTML = O.rows.map(function (r) {
      var mx = Math.max(r.base, r.aware);
      return '<div class="cost"><div class="k">' + r.name + " (" + (r.better === "higher" ? "higher" : "lower") + " is better)</div>" +
        '<div class="row"><span class="from">' + f(r.base) + '</span><span class="arr">→</span><span class="to">' + f(r.aware) + "</span><small>" + r.unit + "</small></div>" +
        '<div class="d">OpenVLA-OFT → w/ AWARE, ' + (r.delta_pct > 0 ? "+" : "−") + f(Math.abs(r.delta_pct)) + "%</div>" +
        '<div class="bar"><i style="width:' + (r.base / mx * 100) + "%;background:" + col.baseLight + '"></i></div>' +
        '<div class="bar" style="margin-top:6px"><i style="width:' + (r.aware / mx * 100) + "%;background:" + col.AWARE + '"></i></div></div>';
    }).join("");
  }

  SITE.inits.push(ablation, controls, alpha, cost);
})();
