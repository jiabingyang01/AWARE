/* Hero facts and key-result cards. Every number is read from PAPER_DATA. */
(function () {
  "use strict";
  var D = SITE.D, col = SITE.colors;

  function heroFacts() {
    var ul = document.getElementById("hero-facts"); if (!ul) return;
    var B = D.setup.backbones;
    var facts = [
      ["<b>" + B.length + "</b> VLA backbones"],
      ["<b>0.5B–7B</b> parameters"],
      ["<b>LIBERO · SIMPLER · CALVIN</b>"],
      ["<b>No</b> training or new parameters"]
    ];
    ul.innerHTML = facts.map(function (f) { return "<li>" + f[0] + "</li>"; }).join("");
  }

  function lib(name) { return D.libero.models.filter(function (m) { return m.name === name; })[0]; }
  function card(k) {
    var ref = (k.refs || []).map(function (x) { return '<span><i style="background:' + x.c + '"></i>' + x.t + "</span>"; }).join("");
    return SITE.h("a", { class: "card kr reveal", href: k.link },
      '<span class="scope">' + SITE.modelHTML(SITE.esc(k.scope)) + "</span>" +
      '<span class="claim">' + SITE.esc(k.title) + "</span>" +
      '<span class="num">' + k.value + (k.unit ? "<small>" + SITE.esc(k.unit) + "</small>" : "") + "</span>" +
      '<span class="lbl">' + k.label + "</span>" +
      (ref ? '<span class="ref">' + ref + "</span>" : "") +
      '<span class="note">' + k.note + "</span>" +
      '<span class="go">' + SITE.esc(k.go) + "</span>");
  }

  function keyResults() {
    var root = document.getElementById("key-results"); if (!root) return;
    var L = D.libero, iLong = L.columns.indexOf("Long");
    var ov = lib("OpenVLA"), p0 = lib("π0"), p05 = lib("π0.5"), oft = lib("OpenVLA-OFT");
    var G = D.simpler.blocks[0], iApple = G.columns.indexOf("Open Top Drawer and Place Apple"), iAvg = G.columns.length - 1;
    var CV = D.calvin, O = D.overhead.rows.filter(function (r) { return r.key === "per_step"; })[0];
    var TP = D.overhead.rows.filter(function (r) { return r.key === "throughput"; })[0];
    var f = SITE.fmt;
    [
      { scope: "OpenVLA · LIBERO-Long", title: "Largest gain on the longest suite", value: SITE.signed(ov.delta[iLong]), unit: "points",
        label: f(ov.base[iLong]) + " → " + f(ov.aware[iLong]) + " success",
        refs: [{ c: col.Base, t: "Base " + f(ov.base[iLong]) }, { c: col.AWARE, t: "AWARE " + f(ov.aware[iLong]) }],
        note: "Long gains the most for every backbone: " + SITE.signed(p0.delta[iLong]) + " for " + SITE.modelHTML("π0") + ", " + SITE.signed(p05.delta[iLong]) + " for " + SITE.modelHTML("π0.5") + ", and OpenVLA-OFT reaches " + f(oft.aware[iLong]) + ".",
        go: "LIBERO results", link: "#results" },
      { scope: "CogACT · SIMPLER Google Robot", title: "Multi-stage drawer-and-apple task", value: SITE.signed(G.delta[iApple]), unit: "points",
        label: "Open Top Drawer and Place Apple, " + f(G.base[iApple]) + " → " + f(G.aware[iApple]),
        note: "The average over the four Google Robot tasks rises from " + f(G.base[iAvg]) + " to " + f(G.aware[iAvg]) + ".",
        go: "SIMPLER results", link: "#results" },
      { scope: "LLaVA-VLA (0.5B) · CALVIN", title: "Gains grow with chain depth", value: f(CV.avg_len.aware, 2), unit: "",
        label: "average completed length, from " + f(CV.avg_len.base, 2),
        note: "Complete five-subtask chains rise from " + f(CV.base[4]) + "% to " + f(CV.aware[4]) + "%.",
        go: "CALVIN results", link: "#results" },
      { scope: "OpenVLA-OFT · Table VI", title: "Small inference overhead", value: "+" + f(D.overhead.added_ms_per_step), unit: "ms",
        label: "per environment step, " + f(O.base) + " → " + f(O.aware) + " ms",
        note: "Two matrix products per layer on the " + D.setup.OFT.N_a + " action elements; throughput stays at " + f(TP.aware) + " Hz.",
        go: "Overhead", link: "#cost-card" }
    ].forEach(function (k) { root.appendChild(card(k)); });
  }

  SITE.inits.push(heroFacts, keyResults);
})();
