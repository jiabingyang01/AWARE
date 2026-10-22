/* Minimal hand-written SVG chart toolkit: scales, axes, tooltip, resize, markers, legends. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var C = SITE.charts = {};

  C.el = function (tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  C.text = function (parent, x, y, str, attrs) {
    var t = C.el("text", Object.assign({ x: x, y: y }, attrs || {}), parent);
    t.textContent = str;
    return t;
  };

  C.linear = function (d0, d1, r0, r1) {
    var f = function (v) { return r0 + (v - d0) / (d1 - d0) * (r1 - r0); };
    f.invert = function (p) { return d0 + (p - r0) / (r1 - r0) * (d1 - d0); };
    f.domain = [d0, d1]; f.range = [r0, r1];
    return f;
  };
  C.band = function (n, r0, r1, pad) {   /* centre of the i-th of n equal bands */
    var bw = (r1 - r0) / n;
    var f = function (i) { return r0 + bw * (i + 0.5); };
    f.bw = bw; f.inner = bw * (1 - (pad == null ? 0.2 : pad));
    return f;
  };

  C.ticks = function (min, max, count) {
    var span = max - min;
    if (span <= 0) return [min];
    var step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = span / count / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [], v = Math.ceil(min / step - 1e-9) * step;
    for (; v <= max + 1e-9; v += step) out.push(Math.round(v / step) * step);
    return out;
  };
  C.decimalsFor = function (ticks) {
    var d = 0;
    ticks.forEach(function (t) { var s = String(+t.toFixed(6)); var i = s.indexOf("."); if (i >= 0) d = Math.max(d, s.length - i - 1); });
    return Math.min(d, 3);
  };

  /* create an svg sized to the container */
  C.svg = function (container, w, h, label) {
    container.innerHTML = "";
    var s = C.el("svg", { width: w, height: h, viewBox: "0 0 " + w + " " + h, role: "img" }, container);
    if (label) s.setAttribute("aria-label", label);
    return s;
  };

  /* y axis with gridlines */
  C.yAxis = function (g, y, x0, x1, ticks, opts) {
    opts = opts || {};
    var ax = C.el("g", { class: "axis" }, g), grid = C.el("g", { class: "grid" }, g);
    var d = opts.decimals != null ? opts.decimals : C.decimalsFor(ticks);
    ticks.forEach(function (t) {
      var yy = Math.round(y(t)) + 0.5;
      if (opts.grid !== false) C.el("line", { x1: x0, x2: x1, y1: yy, y2: yy }, grid);
      C.text(ax, x0 - 8, yy + 4, (opts.format ? opts.format(t) : t.toFixed(d)), { "text-anchor": "end" });
    });
    if (opts.zero != null) C.el("line", { x1: x0, x2: x1, y1: Math.round(y(opts.zero)) + 0.5, y2: Math.round(y(opts.zero)) + 0.5, stroke: "rgba(23,22,28,.35)", "stroke-width": 1 }, g);
    if (opts.title) {
      var mid = (y.range[0] + y.range[1]) / 2;
      C.text(ax, 0, 0, opts.title, { class: "axis-title", transform: "translate(" + (opts.titleX != null ? opts.titleX : 12) + "," + mid + ") rotate(-90)", "text-anchor": "middle" });
    }
    return ax;
  };
  C.xAxis = function (g, x, yPos, ticks, opts) {
    opts = opts || {};
    var ax = C.el("g", { class: "axis" }, g), grid = C.el("g", { class: "grid" }, g);
    ticks.forEach(function (t, i) {
      var xx = Math.round(x(t)) + 0.5;
      if (opts.grid) C.el("line", { x1: xx, x2: xx, y1: opts.gridTop, y2: yPos }, grid);
      var lab = opts.format ? opts.format(t, i) : String(t);
      if (lab !== "") C.text(ax, xx, yPos + 18, lab, { "text-anchor": "middle" });
    });
    if (opts.domain !== false) C.el("line", { class: "domain", x1: x.range[0], x2: x.range[1], y1: Math.round(yPos) + 0.5, y2: Math.round(yPos) + 0.5 }, g);
    if (opts.title) C.text(ax, (x.range[0] + x.range[1]) / 2, yPos + 36, opts.title, { class: "axis-title", "text-anchor": "middle" });
    return ax;
  };

  /* tooltip */
  var tipEl = null, tipAnchor = null;
  C.tip = {
    show: function (html, evt, anchor) {
      if (!tipEl) tipEl = document.getElementById("tip");
      tipAnchor = anchor || (evt && (evt.currentTarget || evt.target)) || null;
      tipEl.innerHTML = html;
      tipEl.classList.add("on");
      var x, y;
      if (evt && evt.clientX != null && !(evt.type && evt.type.indexOf("focus") === 0)) { x = evt.clientX; y = evt.clientY; }
      else if (anchor) { var r = anchor.getBoundingClientRect(); x = r.left + r.width / 2; y = r.top; }
      else { x = window.innerWidth / 2; y = window.innerHeight / 2; }
      var tw = tipEl.offsetWidth, th = tipEl.offsetHeight;
      var left = x + 14, top = y - th - 12;
      if (left + tw > window.innerWidth - 8) left = x - tw - 14;
      if (left < 8) left = 8;
      if (top < 8) top = y + 16;
      tipEl.style.left = left + "px"; tipEl.style.top = top + "px";
    },
    hide: function () { if (tipEl) tipEl.classList.remove("on"); tipAnchor = null; }
  };
  /* a fixed tooltip must never outlive its anchor */
  function tipOn() { return tipEl && tipEl.classList.contains("on"); }
  document.addEventListener("pointermove", function (e) {
    if (!tipOn()) return;
    if (!tipAnchor || !tipAnchor.isConnected || !(tipAnchor === e.target || tipAnchor.contains(e.target))) C.tip.hide();
  }, { passive: true });
  window.addEventListener("scroll", function () { if (tipOn()) C.tip.hide(); }, { passive: true, capture: true });
  window.addEventListener("resize", function () { if (tipOn()) C.tip.hide(); });
  document.addEventListener("pointerdown", function (e) {
    if (tipOn() && !(tipAnchor && (tipAnchor === e.target || tipAnchor.contains(e.target)))) C.tip.hide();
  }, true);
  C.row = function (label, value, color) {
    var sw = color ? '<i class="sw" style="background:' + color + '"></i>' : "";
    return '<div class="tr"><span>' + sw + "<span>" + label + "</span></span><b>" + value + "</b></div>";
  };
  /* attach hover + keyboard focus tooltips to an svg element */
  C.hover = function (node, htmlFn, onEnter, onLeave) {
    node.setAttribute("tabindex", "0");
    var enter = function (e) { C.tip.show(htmlFn(), e, node); if (onEnter) onEnter(); };
    var leave = function () { C.tip.hide(); if (onLeave) onLeave(); };
    node.addEventListener("pointerenter", enter);
    node.addEventListener("pointermove", function (e) { C.tip.show(htmlFn(), e, node); });
    node.addEventListener("pointerleave", leave);
    node.addEventListener("focus", enter);
    node.addEventListener("blur", leave);
  };

  /* re-render on container width change */
  C.responsive = function (container, render) {
    var lastW = -1, raf = null;
    function go() {
      var w = Math.floor(container.clientWidth);
      if (w <= 0 || w === lastW) return;
      lastW = w;
      render(w);
    }
    go();
    if ("ResizeObserver" in window) {
      new ResizeObserver(function () {
        if (raf) cancelAnimationFrame(raf);
        raf = requestAnimationFrame(go);
      }).observe(container);
    } else {
      window.addEventListener("resize", go);
    }
    return { redraw: function () { lastW = -1; go(); } };
  };

  /* polyline or smooth path through points */
  C.path = function (pts, smooth) {
    if (!pts.length) return "";
    if (!smooth || pts.length < 3) return "M" + pts.map(function (p) { return p[0].toFixed(2) + "," + p[1].toFixed(2); }).join("L");
    var d = "M" + pts[0][0].toFixed(2) + "," + pts[0][1].toFixed(2);
    for (var i = 0; i < pts.length - 1; i++) {
      var p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
      var t = 0.16;
      var c1x = p1[0] + (p2[0] - p0[0]) * t, c1y = p1[1] + (p2[1] - p0[1]) * t;
      var c2x = p2[0] - (p3[0] - p1[0]) * t, c2y = p2[1] - (p3[1] - p1[1]) * t;
      d += "C" + c1x.toFixed(2) + "," + c1y.toFixed(2) + " " + c2x.toFixed(2) + "," + c2y.toFixed(2) + " " + p2[0].toFixed(2) + "," + p2[1].toFixed(2);
    }
    return d;
  };

  /* marker shapes */
  C.marker = function (parent, shape, x, y, r, attrs) {
    var a = Object.assign({}, attrs || {});
    if (shape === "triangle") {
      a.d = "M" + x + "," + (y - r * 1.3) + "L" + (x + r * 1.2) + "," + (y + r * 0.85) + "L" + (x - r * 1.2) + "," + (y + r * 0.85) + "Z";
      return C.el("path", a, parent);
    }
    if (shape === "diamond") {
      a.d = "M" + x + "," + (y - r * 1.25) + "L" + (x + r * 1.25) + "," + y + "L" + x + "," + (y + r * 1.25) + "L" + (x - r * 1.25) + "," + y + "Z";
      return C.el("path", a, parent);
    }
    if (shape === "square") {
      a.x = x - r * 0.95; a.y = y - r * 0.95; a.width = r * 1.9; a.height = r * 1.9; a.rx = 1.5;
      return C.el("rect", a, parent);
    }
    a.cx = x; a.cy = y; a.r = r;
    return C.el("circle", a, parent);
  };

  /* html table builder: cols [{key,label,cls}], rows [obj] */
  C.table = function (tableEl, cols, rows, rowCls) {
    var h = "<thead><tr>" + cols.map(function (c) { return '<th class="' + (c.cls || "") + '" scope="col">' + c.label + "</th>"; }).join("") + "</tr></thead><tbody>";
    rows.forEach(function (r, i) {
      h += '<tr class="' + (rowCls ? rowCls(r, i) : "") + '">' + cols.map(function (c) {
        var v = typeof c.key === "function" ? c.key(r) : r[c.key];
        return '<td class="' + (c.cls || "") + '">' + (v == null ? "–" : v) + "</td>";
      }).join("") + "</tr>";
    });
    tableEl.innerHTML = h + "</tbody>";
  };

  /* legend builder: items [{label, color, kind: 'sw'|'line'|'dash'|'dot'|'hollow'|'dia'|'sq'|'tri'|'band'}] */
  C.legend = function (root, items) {
    if (!root) return;
    root.innerHTML = items.map(function (it) {
      var k = it.kind || "", st = "";
      if (k === "line" || k === "dash") st = "color:" + it.color;
      else if (k === "hollow" || k === "tri") st = it.color ? "border-color:" + it.color : "";
      else if (it.color) st = "background:" + it.color;
      return '<span class="it"><i class="sw ' + k + '" style="' + st + '"></i><span>' + SITE.pretty(it.label) + "</span></span>";
    }).join("");
  };

  /* model name for SVG text: π0 / π0.5 with subscript tspans */
  C.modelSVG = function (m) {
    return SITE.esc(m).replace(/π0\.5/g, 'π<tspan baseline-shift="sub" font-size="72%">0.5</tspan>').replace(/π0(?![.\d])/g, 'π<tspan baseline-shift="sub" font-size="72%">0</tspan>');
  };

  /* colour helpers */
  C.mix = function (a, b, t) {
    var pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
    var r = Math.round(((pa >> 16) & 255) * (1 - t) + ((pb >> 16) & 255) * t);
    var g = Math.round(((pa >> 8) & 255) * (1 - t) + ((pb >> 8) & 255) * t);
    var bl = Math.round((pa & 255) * (1 - t) + (pb & 255) * t);
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + bl).toString(16).slice(1);
  };
  C.ramp = function (stops, t) {
    t = Math.max(0, Math.min(1, t));
    var n = stops.length - 1, i = Math.min(n - 1, Math.floor(t * n));
    return C.mix(stops[i], stops[i + 1], t * n - i);
  };
  /* matplotlib viridis, sampled at 9 points */
  C.VIRIDIS = ["#440154", "#472d7b", "#3b528b", "#2c728e", "#21918c", "#28ae80", "#5ec962", "#addc30", "#fde725"];
  C.luma = function (hex) {
    var p = parseInt(hex.slice(1), 16);
    return (0.2126 * ((p >> 16) & 255) + 0.7152 * ((p >> 8) & 255) + 0.0722 * (p & 255)) / 255;
  };
})();
