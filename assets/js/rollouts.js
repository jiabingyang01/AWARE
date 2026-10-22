/* Section 06: paired rollouts (Base vs. w/ AWARE). One tab per benchmark present in window.ROLLOUTS
   (assets/videos/manifest.js); the percentage under each clip is the paper's success rate of that policy in that
   configuration, looked up in PAPER_DATA. */
(function () {
  "use strict";
  var D = SITE.D, T = SITE.T, col = SITE.colors;
  var ORDER = ["OpenVLA", "π0", "π0.5", "OpenVLA-OFT", "CogACT", "SpatialVLA", "LLaVA-VLA"];

  function libRate(ex, rule) {
    var m = D.libero.models.filter(function (x) { return x.name === ex.backbone; })[0];
    var i = D.libero.columns.indexOf(ex.suite);
    return m && i >= 0 ? (rule === "Base" ? m.base[i] : m.aware[i]) : null;
  }
  function simRate(ex, rule) {
    var B = D.simpler.blocks.filter(function (b) { return b.model === ex.backbone; })[0];
    var i = B ? B.columns.indexOf(ex.column) : -1;
    return i >= 0 ? (rule === "Base" ? B.base[i] : B.aware[i]) : null;
  }

  var BENCH = {
    LIBERO: { head: ["LIBERO", "Spatial · Object · Goal · Long"], aspect: "1 / 1", rate: libRate, rate0: 2,
      note: "Percentages are the success rates of each policy on this suite (Table I, 500 episodes per suite)." },
    SIMPLER: { head: ["SIMPLER", "Visual Matching"], aspect: "4 / 3", rate: simRate,
      note: "Percentages are the success rates of each policy on this task (Table II)." },
    CALVIN: { head: ["CALVIN", "ABC→D"], aspect: "1 / 1", rate: null, rate0: 1, note: "" }
  };

  SITE.inits.push(function () {
    var R = window.ROLLOUTS || {}, tabs = document.getElementById("video-tabs"), panels = document.getElementById("video-panels");
    var names = Object.keys(BENCH).filter(function (b) { return (R[b] || []).length; });
    if (!names.length) {
      var sec = document.getElementById("rollouts"); if (sec) sec.hidden = true;
      var link = document.querySelector('#nav-links a[href="#rollouts"]'); if (link) link.parentNode.hidden = true;
      return;
    }
    names.forEach(function (b, i) {
      var id = "v-" + b.toLowerCase();
      tabs.insertAdjacentHTML("beforeend", '<button class="tab" role="tab" id="tab-' + id + '" aria-controls="panel-' + id + '" aria-selected="' + (i === 0) + '"' + (i ? ' tabindex="-1"' : "") + ">" + b + '<span class="cnt">' + R[b].length + "</span></button>");
      panels.insertAdjacentHTML("beforeend", '<div class="tabpanel" role="tabpanel" id="panel-' + id + '" aria-labelledby="tab-' + id + '" tabindex="0"' + (i ? " hidden" : "") + '><div id="' + id + '-root"></div></div>');
    });
    names.forEach(function (b) {
      var cfg = BENCH[b], list = R[b];
      SITE.videoPlayer(document.getElementById("v-" + b.toLowerCase() + "-root"), list, {
        head: cfg.head.concat(["Base · w/ AWARE"]),
        groupTitle: "Backbone",
        groupKey: function (ex) { return ex.backbone; },
        groupOrder: ORDER,
        groupLabel: function (items, key) { return SITE.modelHTML(SITE.esc(key)); },
        rules: function () { return ["Base", "AWARE"]; },
        ruleLabel: function (r) { return r === "AWARE" ? "w/ AWARE" : "Base"; },
        color: function (r) { return r === "AWARE" ? col.AWARE : col.Base; },
        highlight: function (r) { return r === "AWARE"; },
        rate: cfg.rate,
        rateNote: cfg.note,
        rate0: cfg.rate0,
        aspect: cfg.aspect,
        sub: function (ex) {
          return [SITE.modelHTML(SITE.esc(ex.backbone)), ex.suite ? "LIBERO-" + SITE.esc(ex.suite) : "", ex.alpha != null ? "α&nbsp;=&nbsp;" + (+ex.alpha).toFixed(2) : "", ex.episode != null ? (ex.episode_label || "episode") + " " + SITE.esc(ex.episode) : ""].filter(Boolean).join(" · ");
        }
      });
    });
  });
})();
