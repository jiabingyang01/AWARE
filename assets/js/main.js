/* =========================================================================
   SITE CONFIGURATION: edit this object only.
   links: a hero button is shown only for a non-empty entry. Paths may be
   relative or absolute; any URL added here must itself be anonymous.
   ========================================================================= */
window.SITE_CONFIG = {
  links: {
    paper: "",
    code: "",
    video: ""
  }
};

/* ---------------------------------------------------------------------------
   Core helpers shared by all section scripts (one global namespace, no modules,
   so the page also works when opened from file://).
   ------------------------------------------------------------------------- */
(function () {
  "use strict";

  var D = window.PAPER_DATA || {};
  var T = window.PAPER_TEXT || {};
  var FD = window.FIG_DATA || {};

  var SITE = window.SITE = {
    D: D, T: T, FD: FD,
    config: window.SITE_CONFIG,
    inits: [],
    colors: window.SITE_COLORS || {},
    reduced: window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    mq: window.matchMedia ? window.matchMedia("(max-width: 560px)") : { matches: false, addEventListener: function () {} }
  };

  /* path lookup: "D/setup/LIBERO/suites", "T/tldr" */
  SITE.get = function (path) {
    var parts = path.split("/");
    var root = parts.shift();
    var cur = root === "D" ? D : root === "T" ? T : root === "FD" ? FD : undefined;
    for (var i = 0; i < parts.length && cur != null; i++) cur = cur[parts[i]];
    return cur;
  };

  SITE.fmt = function (v, d) {
    if (v == null || isNaN(v)) return "";
    return Number(v).toFixed(d == null ? 1 : d);
  };
  SITE.signed = function (v, d) {
    var s = SITE.fmt(Math.abs(v), d);
    return (v > 0 ? "+" : v < 0 ? "−" : "±") + s;
  };
  SITE.esc = function (s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  };
  /* model display: π0 -> π<sub>0</sub>, π0.5 -> π<sub>0.5</sub> */
  SITE.modelHTML = function (m) {
    return String(m).replace(/π0\.5/g, "π<sub>0.5</sub>").replace(/π0(?![.\d])/g, "π<sub>0</sub>");
  };
  /* escape + typographic touches: π subscripts, sub/superscripts written as _{..} ^{..}, and non-breaking spaces
     so "α = 0.03" or "s = 20" never split across lines */
  SITE.pretty = function (s) {
    return SITE.modelHTML(SITE.esc(s))
      .replace(/_\{([^}]+)\}/g, "<sub>$1</sub>")
      .replace(/\^\{([^}]+)\}/g, "<sup>$1</sup>")
      .replace(/(^|[\s(])([msNHnαℓLK]|N0|N_0) = (?=[\d∞−-])/g, "$1$2 = ")
      .replace(/([msN]) → ∞/g, "$1 → ∞");
  };
  SITE.$ = function (sel, root) { return (root || document).querySelector(sel); };
  SITE.$$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };
  SITE.h = function (tag, attrs, html) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") e.className = attrs[k];
      else if (k === "style") e.style.cssText = attrs[k];
      else if (attrs[k] !== false && attrs[k] != null) e.setAttribute(k, attrs[k]);
    });
    if (html != null) e.innerHTML = html;
    return e;
  };
  SITE.onMQ = function (fn) {
    if (SITE.mq.addEventListener) SITE.mq.addEventListener("change", fn);
    else if (SITE.mq.addListener) SITE.mq.addListener(fn);
  };

  /* segmented control: items [{value,label,dot}], onChange(value) */
  SITE.segmented = function (root, items, value, onChange) {
    root.innerHTML = "";
    var btns = items.map(function (it) {
      var b = SITE.h("button", { type: "button", "aria-pressed": String(it.value === value), "data-v": String(it.value) },
        (it.dot ? '<span class="dot" style="background:' + it.dot + '"></span>' : "") + "<span>" + it.label + "</span>");
      b.addEventListener("click", function () {
        btns.forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        onChange(it.value);
      });
      root.appendChild(b);
      return b;
    });
    return { set: function (v) { btns.forEach(function (x, i) { x.setAttribute("aria-pressed", String(items[i].value === v)); }); } };
  };

  /* range input: keeps the filled part of the track in sync (--pct) */
  SITE.range = function (input, onInput) {
    function paint() {
      var lo = +input.min || 0, hi = +input.max || 100;
      input.style.setProperty("--pct", ((+input.value - lo) / (hi - lo) * 100) + "%");
    }
    input.addEventListener("input", function () { paint(); onInput(+input.value); });
    paint();
    return { paint: paint };
  };

  /* run fn once when el scrolls into view */
  SITE.onVisible = function (el, fn, margin) {
    if (!el) return;
    if (!("IntersectionObserver" in window)) { fn(); return; }
    var io = new IntersectionObserver(function (ents) {
      ents.forEach(function (e) { if (e.isIntersecting) { io.disconnect(); fn(); } });
    }, { rootMargin: margin || "0px 0px -10% 0px" });
    io.observe(el);
  };

  /* right-edge fade on a horizontally scrolling box, shown while more content is hidden */
  SITE.scrollHint = function (sc, extra) {
    if (!sc || sc.__hint) return;
    sc.__hint = true;
    var outer = document.createElement("div");
    outer.className = "sh" + (extra ? " " + extra : "");
    sc.parentNode.insertBefore(outer, sc);
    outer.appendChild(sc);
    var fade = document.createElement("span");
    fade.className = "sh-fade"; fade.setAttribute("aria-hidden", "true");
    outer.appendChild(fade);
    var upd = function () { outer.classList.toggle("more", sc.scrollWidth - sc.clientWidth - sc.scrollLeft > 4); };
    sc.addEventListener("scroll", upd, { passive: true });
    if ("ResizeObserver" in window) new ResizeObserver(upd).observe(sc); else window.addEventListener("resize", upd);
    if ("MutationObserver" in window) new MutationObserver(upd).observe(sc, { childList: true, subtree: true });
    upd();
  };

  /* ------------------------------------------------------------------ boot */
  SITE.boot = function () {
    bindText();
    buildLinks();
    nav();
    lightbox();
    tableToggles();
    figLinks();
    SITE.inits.forEach(function (fn) {
      try { fn(); } catch (err) { if (window.console) console.error(err); }
    });
    tabs();   /* after the section scripts, which may add tab lists of their own */
    SITE.$$(".table-wrap").forEach(function (el) { SITE.scrollHint(el); });
    SITE.$$(".tabs").forEach(function (el) { SITE.scrollHint(el, "tabs-sh"); });
    reveal();
  };

  function bindText() {
    SITE.$$("[data-t]").forEach(function (el) {
      var v = SITE.get("T/" + el.getAttribute("data-t"));
      if (v == null) return;
      el.innerHTML = SITE.pretty(v);
    });
    SITE.$$("[data-d]").forEach(function (el) {
      var v = SITE.get(el.getAttribute("data-d"));
      if (v != null) el.textContent = v;
    });
    SITE.$$("img[data-fig]").forEach(function (img) {
      var f = T.figures && T.figures[img.getAttribute("data-fig")];
      if (f) { img.alt = f.alt || ""; img.setAttribute("data-caption", f.caption || ""); img.setAttribute("data-label", f.group || ""); }
    });
  }

  var ICONS = {
    paper: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M14 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V9z"/><path d="M14 3v6h6M8 13h8M8 17h5"/></svg>',
    code: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M8 7l-5 5 5 5M16 7l5 5-5 5M13.5 4l-3 16"/></svg>',
    video: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="3" y="5" width="14" height="14" rx="2"/><path d="M17 10l4-2.5v9L17 14"/></svg>'
  };

  function buildLinks() {
    var root = SITE.$("#hero-links");
    if (!root) return;
    var L = (SITE.config && SITE.config.links) || {};
    [["paper", "Paper"], ["code", "Code"], ["video", "Video"]].forEach(function (p) {
      var url = (L[p[0]] || "").trim();
      if (!url) return;
      root.appendChild(SITE.h("a", { class: "btn" + (root.children.length ? " ghost" : ""), href: url, target: /^https?:/.test(url) ? "_blank" : null, rel: "noopener noreferrer" }, ICONS[p[0]] + p[1]));
    });
  }

  function nav() {
    var navEl = SITE.$("#nav"), bar = SITE.$("#progress"), toggle = SITE.$("#nav-toggle");
    var links = SITE.$$("#nav-links a");
    var ticking = false;
    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var y = window.scrollY || document.documentElement.scrollTop;
        var max = document.documentElement.scrollHeight - window.innerHeight;
        bar.style.transform = "scaleX(" + (max > 0 ? Math.min(1, y / max) : 0) + ")";
        navEl.classList.toggle("scrolled", y > 12);
        var hero = SITE.$(".hero");
        navEl.classList.toggle("over-hero", !!hero && y < hero.offsetHeight - 70);
        var first = document.getElementById("overview");
        if (first && y < first.offsetTop - window.innerHeight * 0.5) links.forEach(function (a) { a.classList.remove("active"); });
        ticking = false;
      });
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    toggle.addEventListener("click", function () {
      var open = !navEl.classList.contains("open");
      navEl.classList.toggle("open", open);
      toggle.setAttribute("aria-expanded", String(open));
    });
    links.forEach(function (a) {
      a.addEventListener("click", function () {
        navEl.classList.remove("open");
        toggle.setAttribute("aria-expanded", "false");
      });
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && navEl.classList.contains("open")) {
        navEl.classList.remove("open"); toggle.setAttribute("aria-expanded", "false"); toggle.focus();
      }
    });
    if ("IntersectionObserver" in window) {
      var map = {};
      links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
      var io = new IntersectionObserver(function (ents) {
        ents.forEach(function (e) {
          if (e.isIntersecting && map[e.target.id]) {
            links.forEach(function (a) { a.classList.remove("active"); });
            map[e.target.id].classList.add("active");
          }
        });
      }, { rootMargin: "-45% 0px -50% 0px" });
      Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
    }
  }

  /* accessible tabs: any [role=tablist].tabs with [role=tab] children */
  function tabs() {
    SITE.$$('[role="tablist"].tabs').forEach(function (list) {
      var tabsEls = SITE.$$('[role="tab"]', list);
      function select(tab, focus) {
        tabsEls.forEach(function (t) {
          var on = t === tab;
          t.setAttribute("aria-selected", String(on));
          t.tabIndex = on ? 0 : -1;
          var p = document.getElementById(t.getAttribute("aria-controls"));
          if (p) p.hidden = !on;
        });
        if (focus) tab.focus();
        document.dispatchEvent(new CustomEvent("site:tab", { detail: { id: tab.id } }));
      }
      tabsEls.forEach(function (t, i) {
        t.addEventListener("click", function () { select(t); });
        t.addEventListener("keydown", function (e) {
          var k = e.key, n = tabsEls.length, j = null;
          if (k === "ArrowRight") j = (i + 1) % n;
          else if (k === "ArrowLeft") j = (i - 1 + n) % n;
          else if (k === "Home") j = 0;
          else if (k === "End") j = n - 1;
          if (j != null) { e.preventDefault(); select(tabsEls[j], true); }
        });
      });
    });
  }

  function lightbox() {
    var lb = SITE.$("#lightbox"), img = SITE.$("#lb-img"), cap = SITE.$("#lb-cap"), close = SITE.$("#lb-close");
    var last = null;
    SITE.openLightbox = function (src, alt, caption, label) {
      last = document.activeElement;
      img.src = src; img.alt = alt || "";
      img.classList.toggle("svg", /\.svg$/.test(src));
      cap.innerHTML = (label ? "<b>" + SITE.esc(label) + ".</b> " : "") + SITE.pretty(caption || "");
      lb.classList.add("on"); close.focus();
      document.body.style.overflow = "hidden";
    };
    function hide() {
      lb.classList.remove("on"); img.removeAttribute("src");
      document.body.style.overflow = "";
      if (last) last.focus();
    }
    close.addEventListener("click", hide);
    lb.addEventListener("click", function (e) { if (e.target === lb) hide(); });
    document.addEventListener("keydown", function (e) { if (e.key === "Escape" && lb.classList.contains("on")) hide(); });
    SITE.zoomable = function (im) {
      im.tabIndex = 0;
      im.setAttribute("role", "button");
      var open = function () {
        var src = im.getAttribute("data-full") || im.currentSrc || im.src;
        SITE.openLightbox(src, im.alt, im.getAttribute("data-caption"), im.getAttribute("data-label"));
      };
      im.addEventListener("click", open);
      im.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } });
    };
    SITE.$$("img[data-zoom]").forEach(SITE.zoomable);
  }

  /* "Figure N" badge on a card: opens the paper's original figure */
  function figLinks() {
    SITE.$$("[data-fig-open]").forEach(function (b) {
      var paint = function () {
        var F = T.figures && T.figures[b.getAttribute("data-fig-open")];
        if (!F) { b.hidden = true; return; }
        b.hidden = false;
        b.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="18" height="18" rx="3"/><path d="M8 15l3-4 3 3 2-2"/></svg>' +
          '<span class="fl-txt">' + SITE.esc(F.group).replace("Figure", 'Fig<span class="wide-only">ure</span>') + "</span>";
        b.title = "Open the paper's figure: " + F.title;
        b.setAttribute("aria-label", "Open the paper's figure: " + F.title);
      };
      paint();
      b.addEventListener("click", function () {
        var F = T.figures[b.getAttribute("data-fig-open")];
        if (F) SITE.openLightbox(F.files.svg || F.files.webp, F.alt, F.caption, F.group);
      });
      new MutationObserver(paint).observe(b, { attributes: true, attributeFilter: ["data-fig-open"] });
    });
  }

  function tableToggles() {
    SITE.$$("[data-toggle-table]").forEach(function (b) {
      b.addEventListener("click", function () {
        var t = document.getElementById(b.getAttribute("data-toggle-table"));
        if (!t) return;
        t.hidden = !t.hidden;
        b.setAttribute("aria-expanded", String(!t.hidden));
        b.textContent = t.hidden ? "Show table" : "Hide table";
      });
    });
  }

  function reveal() {
    var els = SITE.$$(".reveal");
    if (SITE.reduced || !("IntersectionObserver" in window)) { els.forEach(function (e) { e.classList.add("in"); }); return; }
    var io = new IntersectionObserver(function (ents) {
      ents.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -6% 0px", threshold: 0.04 });
    els.forEach(function (e) { io.observe(e); });
  }
})();
