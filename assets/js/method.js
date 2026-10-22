/* Section 03: the four steps of the rule, the per-family token mapping, and the injection-weight map computed from
   the Action Entropy of Figure 1b (FIG_DATA.entropy) with w = alpha * l/(L-1) * H. */
(function () {
  "use strict";
  var D = SITE.D, FD = SITE.FD, C = SITE.charts, col = SITE.colors;

  function eqCards() {
    var root = document.getElementById("eq-grid"); if (!root) return;
    var cards = [
      { n: 1, c: "#21918c", step: "Read", title: "Action Entropy of every element",
        eq: [["entropy", "14.679em"]],
        p: "Each action element's hidden state at the output of layer ℓ goes through the frozen language head (the logit lens); the normalized entropy over K = " + D.setup.K + " candidate tokens is its uncertainty, one value per element." },
      { n: 2, c: col.obs, step: "Weigh", title: "Instruction-guided observation weights",
        eq: [["gate", "25.973em"]],
        p: "The instruction tokens score every observation token by their non-negative alignment; tokens at or above the uniform level keep full weight and the rest are attenuated. One matrix product per forward pass, no learned parameters." },
      { n: 3, c: "#a47830", step: "Retrieve", title: "Look up the observation memory",
        eq: [["inj", "17.772em"]],
        p: "The observation tokens are both keys and values; each action element queries them with the hidden state entering its FFN. The scale s matches the retrieved features to the magnitude of the native FFN output." },
      { n: 4, c: col.AWARE, step: "Write", title: "Blend by entropy and depth",
        eq: [["weight", "11.126em"], ["blend", "25.258em"]],
        p: "The weight grows with relative depth and with the element's entropy; the blend replaces that element's FFN output, and every non-action token keeps its native output." }
    ];
    root.innerHTML = cards.map(function (k) {
      return '<div class="card eq-card reveal" style="--c:' + k.c + '"><span class="step"><i>' + k.n + "</i>" + k.step + "</span><h3>" + k.title + "</h3>" +
        '<div class="eq-panel"><div>' + k.eq.map(function (e) {
          return '<div class="eq sm"><img src="assets/img/eq/' + e[0] + '.svg" style="--w:' + e[1] + '" alt="" decoding="async"></div>';
        }).join("") + "</div></div><p>" + SITE.pretty(k.p) + "</p></div>";
    }).join("");
    SITE.$$(".eq-card img", root).forEach(function (im) { im.style.maxWidth = "100%"; });
  }

  function mapping() {
    var t = document.getElementById("map-table"); if (!t) return;
    var rows = [
      ["Parallel chunk decoding", "OpenVLA-OFT", "all action placeholder tokens, read together in one bidirectional pass", '<span class="na">N<sub>a</sub> = ' + D.setup.OFT.N_a + "</span><small>" + D.setup.OFT.H + " steps × " + D.setup.OFT.D_a + " dims</small>"],
      ["Causal autoregressive", "OpenVLA, SpatialVLA", "each newly decoded action token", '<span class="na">N<sub>a</sub> = 1</span><small>per decoding step</small>'],
      ["Dual-system", "CogACT", "the cognition token that conditions the action decoder", '<span class="na">1 token</span>'],
      ["Flow-matching expert", SITE.modelHTML("π0, π0.5"), "the instruction positions of the prefix that condition the expert, which has no vocabulary head", '<span class="na">instruction</span>']
    ];
    t.innerHTML = '<thead><tr><th scope="col">Family</th><th scope="col">Entropy read at</th><th scope="col">Elements</th></tr></thead><tbody>' +
      rows.map(function (r) { return "<tr><td>" + r[0] + "<small>" + r[1] + "</small></td><td>" + r[2] + "</td><td>" + r[3] + "</td></tr>"; }).join("") + "</tbody>";
  }

  /* ---------------------------------------------------------- injection-weight map */
  function weightMap() {
    var E = FD.entropy, el = document.getElementById("wm-chart");
    if (!E || !el) return;
    var A = D.ablation, AL = FD.alpha && FD.alpha["OpenVLA-OFT"];
    var L = E.H.length, NE = E.H[0].length;
    var rowMean = E.H.map(function (r) { return r.reduce(function (a, b) { return a + b; }, 0) / r.length; });
    var VARIANTS = [
      { key: "full", label: "AWARE", dot: col.AWARE, f: function (l, n, a) { return a * l / (L - 1) * E.H[l][n]; },
        formula: "<b>Full rule.</b> <span class=\"m\"><i>w</i>&nbsp;=&nbsp;<i>α</i>&nbsp;·&nbsp;ℓ/(<i>L</i>−1)&nbsp;·&nbsp;𝓗<sub><i>n</i></sub><sup>(ℓ)</sup></span>: deep layers and still-uncertain elements receive the most evidence." },
      { key: "no_element", label: "Uniform element weights", dot: "#21918c", f: function (l, n, a) { return a * l / (L - 1) * rowMean[l]; },
        formula: "<b>Uniform element weights.</b> Each element gets the chunk-average entropy of its layer, so a whole row shares one weight and the far steps lose their extra share." },
      { key: "no_depth", label: "No depth factor", dot: "#a47830", f: function (l, n, a) { return a * E.H[l][n]; },
        formula: "<b>No depth factor.</b> <span class=\"m\"><i>w</i>&nbsp;=&nbsp;<i>α</i>&nbsp;·&nbsp;𝓗</span>: shallow layers, where the observation is still well represented, receive as much evidence as deep ones." },
      { key: "no_instruction", label: "No instruction guidance", dot: col.obs, f: function (l, n, a) { return a * l / (L - 1) * E.H[l][n]; },
        formula: "<b>No instruction guidance.</b> The weights are those of the full rule; what changes is the retrieved content, which no longer down-weights the observation tokens the instruction attends to least." },
      { key: "base", label: "Base policy", dot: col.Base, f: function () { return 0; },
        formula: "<b>Base policy.</b> No retrieval: every FFN output is the native one." }
    ];
    var state = { v: "full", alpha: 0.03 };
    var scaleEl = document.getElementById("wm-scale");
    /* colour scale: the largest weight of the full rule at the largest strength (0.10), square-root mapped so the
       map visibly brightens with the slider; weights above it (possible without the depth factor) saturate */
    var VMAX = 0;
    for (var l0 = 0; l0 < L - 1; l0++) for (var n0 = 0; n0 < NE; n0++) VMAX = Math.max(VMAX, 0.1 * l0 / (L - 1) * E.H[l0][n0]);
    var tone = function (wv) { return C.ramp(RAMP, Math.sqrt(Math.max(0, wv) / VMAX)); };
    if (scaleEl) scaleEl.innerHTML = "w&nbsp;=&nbsp;0 <i></i> " + VMAX.toFixed(3) + "+";
    var RAMP = ["#f7f5fb", "#e6d8f2", "#c7a9e3", "#a074c0", "#7030a0", "#3d1263"];

    var pre = document.getElementById("wm-presets");
    VARIANTS.forEach(function (v) {
      var b = SITE.h("button", { class: "preset", type: "button", "aria-pressed": String(v.key === state.v) }, '<span class="dot" style="background:' + v.dot + '"></span>' + v.label);
      b.addEventListener("click", function () {
        SITE.$$(".preset", pre).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        state.v = v.key; update();
      });
      pre.appendChild(b);
    });
    var slider = document.getElementById("wm-alpha"), out = document.getElementById("wm-alpha-out");
    var rng = SITE.range(slider, function (v) { state.alpha = v / 100; update(); });

    function variant() { return VARIANTS.filter(function (v) { return v.key === state.v; })[0]; }
    var resp = C.responsive(el, function (w) {
      var V = variant(), a = state.alpha;
      var m = { l: 40, r: 8, t: 8, b: 40 }, h = Math.max(260, Math.min(360, w * 0.56));
      var svg = C.svg(el, w, h, "Injection weight per layer and chunk element");
      var cw = (w - m.l - m.r) / NE, ch = (h - m.t - m.b) / L;
      var g = C.el("g", null, svg);
      /* row r shows the weight applied at layer r = l + 1, read from the entropy of layer l */
      for (var r = 0; r < L; r++) {
        for (var n = 0; n < NE; n++) {
          var wv = r === 0 ? 0 : V.f(r - 1, n, a);
          C.el("rect", { x: m.l + n * cw, y: h - m.b - (r + 1) * ch, width: Math.ceil(cw) + 0.4, height: Math.ceil(ch) + 0.4, fill: tone(wv) }, g);
        }
      }
      for (var k = 1; k < E.steps; k++) C.el("line", { x1: m.l + k * E.dims * cw, x2: m.l + k * E.dims * cw, y1: m.t, y2: h - m.b, stroke: "rgba(255,255,255,.55)", "stroke-width": 1 }, g);
      var ax = C.el("g", { class: "axis" }, g);
      for (var s = 0; s < E.steps; s++) C.text(ax, m.l + (s * E.dims + E.dims / 2) * cw, h - m.b + 16, String(s + 1), { "text-anchor": "middle" });
      C.text(ax, (m.l + w - m.r) / 2, h - 6, "future step of the chunk (7 dims each)", { class: "axis-title", "text-anchor": "middle" });
      [1, 8, 16, 24, 31].forEach(function (t) { C.text(ax, m.l - 7, h - m.b - (t + 0.5) * ch + 4, String(t), { "text-anchor": "end" }); });
      C.text(ax, 0, 0, "layer receiving evidence", { class: "axis-title", transform: "translate(11," + ((m.t + h - m.b) / 2) + ") rotate(-90)", "text-anchor": "middle" });
      var hover = C.el("rect", { fill: "none", stroke: "#16141d", "stroke-width": 1.5, width: cw, height: ch, opacity: 0, "pointer-events": "none" }, g);
      var hit = C.el("rect", { x: m.l, y: m.t, width: w - m.l - m.r, height: h - m.t - m.b, class: "hit" }, svg);
      var cur = { r: 24, n: 52 };
      function at(evt) {
        if (evt && evt.clientX != null) {
          var rc = svg.getBoundingClientRect();
          cur.n = Math.max(0, Math.min(NE - 1, Math.floor((evt.clientX - rc.left - m.l) / cw)));
          cur.r = Math.max(0, Math.min(L - 1, Math.floor((h - m.b - (evt.clientY - rc.top)) / ch)));
        }
        hover.setAttribute("x", m.l + cur.n * cw); hover.setAttribute("y", h - m.b - (cur.r + 1) * ch); hover.setAttribute("opacity", 1);
        var wv = cur.r === 0 ? 0 : V.f(cur.r - 1, cur.n, a), Hv = cur.r === 0 ? null : E.H[cur.r - 1][cur.n];
        return '<div class="th">Step ' + (Math.floor(cur.n / E.dims) + 1) + " · dim " + (cur.n % E.dims + 1) + " · layer " + cur.r + "</div>" +
          (Hv == null ? C.row("weight", "0 (first layer)") : C.row("entropy of layer " + (cur.r - 1), SITE.fmt(Hv, 3)) + C.row("weight w", SITE.fmt(wv, 4), tone(wv)));
      }
      hit.addEventListener("pointermove", function (e) { C.tip.show(at(e), e, hit); });
      C.hover(hit, function () { return at(null); }, null, function () { hover.setAttribute("opacity", 0); });
    });

    function suiteBars(values, ref, lo, hi, n) {
      var S = A.columns.slice(0, n || A.columns.length);
      return '<div class="wm-suites">' + S.map(function (s, i) {
        var v = values[i], p = (v - lo) / (hi - lo) * 100, q = ref ? (ref[i] - lo) / (hi - lo) * 100 : null;
        return '<div class="wm-suite"><span>' + s + '</span><div class="tr"><i style="width:' + Math.max(0, Math.min(100, p)) + '%"></i>' +
          (q != null ? '<u style="left:calc(' + Math.max(0, Math.min(100, q)) + '% - 1px)" title="Base"></u>' : "") + "</div><b>" + SITE.fmt(v) + "</b></div>";
      }).join("") + "</div>";
    }

    function update() {
      var V = variant(), a = state.alpha;
      out.textContent = a.toFixed(2);
      slider.value = String(Math.round(a * 100)); rng.paint();
      slider.disabled = state.v === "base";
      var sum = 0, mx = 0;
      for (var l = 0; l < L - 1; l++) for (var n = 0; n < NE; n++) { var wv = V.f(l, n, a); sum += wv; mx = Math.max(mx, wv); }
      var mean = sum / ((L - 1) * NE);
      document.getElementById("wm-mean").textContent = mean.toFixed(4);
      document.getElementById("wm-max").innerHTML = "largest single weight " + mx.toFixed(3);
      document.getElementById("wm-bar").style.width = Math.min(100, Math.sqrt(mean / VMAX) * 100) + "%";
      document.getElementById("wm-formula").innerHTML = V.formula;
      var S = document.getElementById("wm-success"), base = A.rows[0].values, iAvg = A.columns.length - 1;
      if (V.key === "full" && AL) {
        var ia = AL.alphas.map(function (x) { return Math.round(x * 100); }).indexOf(Math.round(a * 100));
        var vals = A.columns.slice(0, 4).map(function (s) { return AL.suites[s][ia]; });
        var avg = vals.reduce(function (x, y) { return x + y; }, 0) / vals.length;
        var tuned = A.columns.slice(0, 4).map(function (s) { return s + " " + AL.best[s].toFixed(2); }).join(" · ");
        S.innerHTML = '<div class="k">Measured success at α = ' + a.toFixed(2) + ' <span class="muted">(Figure 5, OpenVLA-OFT)</span></div>' +
          '<div class="v">' + SITE.fmt(avg, 2) + "<small>% avg</small></div>" +
          '<div class="n">base ' + SITE.fmt(base[iAvg]) + "% · the ticks mark the base policy</div>" +
          suiteBars(vals, base.slice(0, 4), 90, 100, 4) +
          '<div class="d">Table I uses the per-suite strengths ' + tuned + ".</div>";
      } else {
        var row = A.rows.filter(function (r) { return r.key === V.key; })[0];
        S.innerHTML = '<div class="k">Measured success of this variant <span class="muted">(Table IV, strengths of Table I)</span></div>' +
          '<div class="v">' + SITE.fmt(row.values[iAvg]) + "<small>% avg</small></div>" +
          '<div class="n">full rule ' + SITE.fmt(A.rows[A.rows.length - 1].values[iAvg]) + "% · base " + SITE.fmt(base[iAvg]) + "%</div>" +
          suiteBars(row.values, V.key === "base" ? null : base, 90, 100) +
          (V.key === "base" ? "" : '<div class="d">The slider only rescales the map; the success rates are those of Table IV.</div>');
      }
      resp.redraw();
    }
    update();
  }

  SITE.inits.push(eqCards, mapping, weightMap);
})();
