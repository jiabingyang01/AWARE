/* Synchronized rollout player. One card per benchmark: a configuration bar, an optional suite bar, example chips,
   and a row of clips that play in sync (one play / pause / restart / seek / speed control for the whole row).
   Site scripts call SITE.videoPlayer(root, list, opts) with
     opts.groupKey(ex)        -> key of the configuration an example belongs to (default: backbone)
     opts.groupLabel(items)   -> HTML label of that configuration
     opts.groupOrder          -> array of keys in display order (optional)
     opts.rules(ex)           -> ordered clip labels to show for this example
     opts.color(rule, ex)     -> colour of a clip
     opts.rate(ex, rule)      -> the paper's success rate for that clip's configuration, or null
     opts.rateText(v)         -> formatting of the rate
     opts.highlight(rule, ex) -> true for the clip(s) to emphasise
     opts.sub(ex)             -> HTML of the line under the title
   An example may carry `chip`, a shorter label for its button when two examples share a title.
     opts.aspect, opts.rate0 (initial playback speed), opts.syncNote, opts.rateNote */
(function () {
  "use strict";

  function meta(items) {
    return '<div class="vid-meta">' + items.filter(Boolean).map(function (s) { return "<span>" + s + "</span>"; }).join("") + "</div>";
  }
  var SUITE_ORDER = ["Spatial", "Object", "Goal", "Long"];

  SITE.videoPlayer = function (root, list, opts) {
    opts = opts || {};
    var gk = opts.groupKey || function (ex) { return ex.backbone; };
    var groups = [], map = {};
    list.forEach(function (ex) {
      var k = gk(ex);
      if (!map[k]) { map[k] = { key: k, items: [] }; groups.push(map[k]); }
      map[k].items.push(ex);
    });
    if (opts.groupOrder) groups.sort(function (a, b) {
      var ia = opts.groupOrder.indexOf(a.key), ib = opts.groupOrder.indexOf(b.key);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });
    var state = { gi: 0, idx: 0, loaded: false, rate: String(opts.rate0 || 1) };
    var cur = groups[0].items, cfgItems = cur, curSuite = null;

    root.innerHTML = '<div class="vid-head">' + meta(opts.head || []) + "</div>" +
      '<div class="card player">' + (groups.length > 1 ? '<div class="cfg-row"><span class="cfg-lab">' + (opts.groupTitle || "Configuration") + '</span><div class="cfg-bar main-bar" role="group" aria-label="' + (opts.groupTitle || "Configuration") + '"></div></div>' : "") +
      '<div class="cfg-row suite-row" hidden><span class="cfg-lab">Suite</span><div class="cfg-bar suite-bar" role="group" aria-label="Suite"></div></div>' +
      '<div class="ex-chips" role="group" aria-label="Examples"></div><div class="ex-body"></div></div>';
    var bar = root.querySelector(".main-bar"), chips = root.querySelector(".ex-chips"), body = root.querySelector(".ex-body");
    var srow = root.querySelector(".suite-row"), sbar = root.querySelector(".suite-bar");

    function suitesFor(items) {
      var present = SUITE_ORDER.filter(function (su) { return items.some(function (ex) { return ex.suite === su; }); });
      if (!present.length) { srow.hidden = true; cur = items; chipsFor(cur); return; }
      srow.hidden = false; sbar.innerHTML = "";
      if (present.indexOf(curSuite) < 0) curSuite = present[0];
      present.forEach(function (su) {
        var n = items.filter(function (ex) { return ex.suite === su; }).length;
        var b = SITE.h("button", { class: "cfg-btn", type: "button", "aria-pressed": String(su === curSuite) }, su + '<span class="cn">' + n + "</span>");
        b.addEventListener("click", function () {
          SITE.$$(".cfg-btn", sbar).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          curSuite = su; cur = items.filter(function (ex) { return ex.suite === su; }); chipsFor(cur); show(0);
        });
        sbar.appendChild(b);
      });
      cur = items.filter(function (ex) { return ex.suite === curSuite; });
      chipsFor(cur);
    }
    function chipsFor(items) {
      chips.innerHTML = "";
      if (items.length < 2) { chips.style.display = "none"; return; }
      chips.style.display = "";
      items.forEach(function (ex, i) {
        var b = SITE.h("button", { class: "ex-chip", type: "button", "aria-pressed": String(i === 0) }, SITE.esc(ex.chip || ex.title || ex.task));
        b.addEventListener("click", function () {
          SITE.$$(".ex-chip", chips).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          show(i);
        });
        chips.appendChild(b);
      });
    }
    if (bar) {
      groups.forEach(function (g, gi) {
        var b = SITE.h("button", { class: "cfg-btn", type: "button", "aria-pressed": String(gi === 0) },
          (opts.groupLabel ? opts.groupLabel(g.items, g.key) : SITE.modelHTML(SITE.esc(g.key))) + '<span class="cn">' + g.items.length + "</span>");
        b.addEventListener("click", function () {
          SITE.$$(".cfg-btn", bar).forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
          state.gi = gi; cfgItems = g.items; suitesFor(cfgItems); show(0);
        });
        bar.appendChild(b);
      });
    }
    suitesFor(cfgItems);

    var vids = [], raf = null, playing = false, scrub, timeEl, playBtn, loopIn;
    function maxDur() { return vids.reduce(function (m, v) { return Math.max(m, isFinite(v.duration) ? v.duration : 0); }, 0); }
    function master() { var md = maxDur(), mv = vids[0]; vids.forEach(function (v) { if (isFinite(v.duration) && v.duration >= md) mv = v; }); return mv; }
    function fmtT(s) { s = Math.max(0, s || 0); return Math.floor(s / 60) + ":" + ("0" + Math.floor(s % 60)).slice(-2) + "." + Math.floor((s % 1) * 10); }
    function sync() {
      if (!vids.length) return;
      var mv = master(), t = mv.currentTime;
      vids.forEach(function (v) {
        if (v === mv || !isFinite(v.duration)) return;
        var target = Math.min(t, v.duration);
        if (Math.abs(v.currentTime - target) > 0.12) { try { v.currentTime = target; } catch (e) { /* not seekable yet */ } }
        if (playing && v.paused && t < v.duration - 0.05) v.play().catch(function () {});
      });
      var md = maxDur();
      if (scrub && md) { scrub.value = String(t / md * 1000); scrub.style.setProperty("--pct", (t / md * 100) + "%"); }
      if (timeEl) timeEl.textContent = fmtT(t) + " / " + fmtT(md);
    }
    function loop() { sync(); if (playing) raf = requestAnimationFrame(loop); }
    function setPlay(p) {
      playing = p;
      if (playBtn) {
        playBtn.setAttribute("aria-label", p ? "Pause all clips" : "Play all clips");
        playBtn.innerHTML = (p ? '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>Pause' : '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 5l12 7-12 7z"/></svg>Play');
      }
      if (p && !vids.length) { playing = false; return; }
      if (p) {
        var md = maxDur();
        if (md && master().currentTime >= md - 0.05) seek(0);
        vids.forEach(function (v) { if (!isFinite(v.duration) || v.currentTime < v.duration - 0.05) v.play().catch(function () {}); });
        cancelAnimationFrame(raf); raf = requestAnimationFrame(loop);
      } else {
        vids.forEach(function (v) { v.pause(); });
        cancelAnimationFrame(raf); sync();
      }
    }
    function seek(t) { vids.forEach(function (v) { try { v.currentTime = Math.min(t, isFinite(v.duration) ? v.duration : t); } catch (e) { /* ignore */ } }); sync(); }
    function load() {
      if (state.loaded) return;
      state.loaded = true;
      vids.forEach(function (v) { if (v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; } });
    }

    function show(i) {
      setPlay(false);
      state.idx = i;
      var ex = cur[i];
      var order = opts.rules ? opts.rules(ex) : (ex.clips || []).map(function (c) { return c.rule; });
      var clips = order.map(function (r) { return (ex.clips || []).filter(function (c) { return c.rule === r; })[0]; }).filter(Boolean);
      var n = clips.length, anyRate = false;
      var tiles = clips.map(function (c) {
        var ok = c.success === true, bad = c.success === false, v = opts.rate ? opts.rate(ex, c.rule) : null;
        if (v != null) anyRate = true;
        var rt = v == null ? "" : '<span class="rt">' + (opts.rateText ? opts.rateText(v) : (+v).toFixed(1) + "%") + "</span>";
        var badge = ok || bad ? '<span class="badge ' + (ok ? "ok" : "fail") + '">' + (ok ? "✓ Success" : "✕ Failure") + "</span>" : "";
        var label = opts.ruleLabel ? opts.ruleLabel(c.rule, ex) : SITE.esc(c.rule);
        return '<div class="tile live' + (opts.highlight && opts.highlight(c.rule, ex) ? " hl" : "") + '" style="--c:' + (opts.color ? opts.color(c.rule, ex) : "var(--acc)") + '"><div class="scr">' +
          badge + '<video muted playsinline preload="none" ' + (c.poster ? 'poster="' + SITE.esc(c.poster) + '" ' : "") + 'data-src="' + SITE.esc(c.file) + '" aria-label="' + SITE.esc(String(c.rule).replace(/<[^>]+>/g, "") + " rollout" + (ok ? ", success" : bad ? ", failure" : "")) + '"></video>' +
          '</div><div class="rl"><i></i>' + label + rt + "</div></div>";
      }).join("");
      var ar = String(opts.aspect || "1").split("/").map(Number), wide = ar.length === 2 && ar[0] / ar[1] > 1.2;
      var cols = n === 2 ? " n2" + (wide ? " wide" : "") : n === 3 ? " n3" : "";
      body.innerHTML = '<div class="ex-title">' + SITE.esc(ex.title || ex.task || "") + "</div>" +
        (ex.instruction ? '<p class="ex-instr">“' + SITE.esc(ex.instruction) + "”</p>" : "") +
        '<p class="ex-sub">' + (opts.sub ? opts.sub(ex) : "") + "</p>" +
        '<div class="tiles live-tiles' + cols + '" style="' + (opts.aspect ? "--ar:" + opts.aspect + ";" : "") + (n > 4 ? "grid-template-columns:repeat(" + n + ",1fr)" : "") + '">' + tiles + "</div>" +
        '<div class="pctl"><button class="icon-btn" data-act="play" aria-label="Play all clips"></button>' +
        '<button class="icon-btn" data-act="restart" aria-label="Restart all clips"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="M3 12a9 9 0 109-9 9 9 0 00-6.4 2.6L3 8"/><path d="M3 3v5h5"/></svg>Restart</button>' +
        '<input type="range" min="0" max="1000" value="0" step="1" aria-label="Seek all clips" style="--pct:0%">' +
        '<span class="time">0:00.0</span>' +
        '<label><input type="checkbox" data-act="loop" checked> loop</label>' +
        '<select class="icon-btn" data-act="rate" aria-label="Playback speed">' + ["0.5", "1", "2", "4"].map(function (r) { return '<option value="' + r + '"' + (r === state.rate ? " selected" : "") + ">" + r + "×</option>"; }).join("") + "</select></div>" +
        '<p class="sync-note">' + (opts.syncNote || "All clips of an example start from the same initial state and play in sync. Clips that finish early hold their last frame.") +
        (anyRate && opts.rateNote ? " " + opts.rateNote : "") + "</p>";
      vids = SITE.$$("video", body);
      playBtn = body.querySelector('[data-act="play"]');
      scrub = body.querySelector('input[type="range"]');
      timeEl = body.querySelector(".time");
      loopIn = body.querySelector('[data-act="loop"]');
      playBtn.addEventListener("click", function () { load(); setPlay(!playing); });
      body.querySelector('[data-act="restart"]').addEventListener("click", function () { load(); seek(0); setPlay(true); });
      scrub.addEventListener("input", function () { load(); seek(+scrub.value / 1000 * maxDur()); });
      body.querySelector('[data-act="rate"]').addEventListener("change", function (e) { state.rate = e.target.value; vids.forEach(function (v) { v.playbackRate = +state.rate; }); });
      vids.forEach(function (v) { v.defaultPlaybackRate = +state.rate; v.playbackRate = +state.rate; });
      vids.forEach(function (v) {
        v.addEventListener("loadedmetadata", sync);
        v.addEventListener("ended", function () {
          var md = maxDur();
          if (v === master() || vids.every(function (x) { return x.ended || (isFinite(x.duration) && x.currentTime >= x.duration - 0.05); })) {
            if (loopIn.checked && playing) { seek(0); vids.forEach(function (x) { x.play().catch(function () {}); }); }
            else if (master().currentTime >= md - 0.05) setPlay(false);
          }
        });
      });
      setPlay(false);
      if (state.loaded) { state.loaded = false; load(); }
    }
    show(0);
    SITE.onVisible(root, function () { load(); }, "200px 0px 200px 0px");
    document.addEventListener("site:tab", function () { if (!root.offsetParent) setPlay(false); });
    return { setPlay: setPlay };
  };
})();
