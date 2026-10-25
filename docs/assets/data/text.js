/* Prose of the AWARE page. The abstract and the figure captions are the paper's; the rest summarises it. */
window.PAPER_TEXT = {
  title: "AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for Vision-Language-Action Models",

  tldr: "Deeper in a VLA's forward pass, the action tokens rely less and less on the observation, and the distant steps of an action chunk stay uncertain the longest. AWARE uses the model's own feed-forward layers as a memory over the encoded observation and blends what each action element retrieves into its FFN output, in proportion to that element's layer-wise Action Entropy and to depth. It needs no training, no extra observation cues and no new modules.",

  abstract: "Vision-Language-Action (VLA) models combine multimodal perception with robotic control, demonstrating remarkable potential for generalizable manipulation. Yet, enhancing their execution typically relies on auxiliary sensory inputs (e.g., depth, 3D point clouds) or costly fine-tuning. Inspired by the insight that Feed-Forward Networks (FFNs) act as key-value memory, we propose Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence (AWARE), an effective, training-free, and plug-and-play module for VLAs. Specifically, AWARE tracks layer-wise uncertainty via an element-wise Action Entropy metric, and retrieves observation evidence into the element's FFN through associative memory lookup. Crucially, the retrieved features are weighted by instruction-guided cross-modal attention to isolate task-relevant observation cues, while injection strength is adaptively scaled by the element's entropy and relative network depth. This selectively channels observation evidence to high-entropy action dimensions and deep layers where grounding decays, while leaving converged representations intact. Across LIBERO, SIMPLER, CALVIN, and real-world robot manipulation, AWARE consistently improves diverse VLA architectures with minimal computational overhead, achieving the largest gains on long-horizon tasks without architectural changes or additional training.",

  diagnosis: {
    lede: "Looking inside OpenVLA-OFT on LIBERO during one forward pass, two things stand out. The share of the action representation explained by the observation tokens falls with depth, to about half by layer 26 in every suite. And the elements of an action chunk do not settle together: near steps become confident in the middle of the stack, while distant steps stay uncertain for several more layers, and the entropy gap between the first and the last step peaks at 0.25 around layer 17. Both point to the same remedy: give observation evidence back where the action is still undecided.",
    footprint_sub: "OpenVLA-OFT, one curve per LIBERO suite; by layer 26 the share is about half of its initial value in all four.",
    entropy_sub: "OpenVLA-OFT: 8 steps × 7 dimensions = 56 chunk elements, read at the output of every layer; the dashed line is the isocontour that traces the decision wavefront."
  },

  method: {
    lede: "A feed-forward layer is a soft key-value lookup: FFN(h) = Σ_{i} φ(⟨h, k_{i}⟩) v_{i}. AWARE adds one more memory to that lookup, the encoded observation tokens, used as both keys and values. Every action element queries it with its own hidden state, and the answer replaces part of that element's FFN output: more for elements that are still uncertain, and more in deeper layers.",
    mapping_sub: "The rule is the same for every backbone; only the positions whose entropy is read depend on how the family forms its action.",
    widget_sub: "The map uses the Action Entropy of Figure 1b for OpenVLA-OFT. Choose a variant of the rule and a strength to see which elements and layers receive observation evidence; the panel on the right gives the measured success of that variant."
  },

  results: {
    lede: "AWARE is evaluated on seven VLA backbones from 0.5B to 7B parameters, four single-system and three dual-system, on LIBERO, SIMPLER and CALVIN. Base policies and AWARE share the official checkpoints, environments, initial states and pipelines; the strength α ∈ [0.01, 0.10] is set per task suite.",
    calvin_sub: "LLaVA-VLA (0.5B), 500 language-conditioned chains of five subtasks; one strength α = 0.01 for the whole benchmark.",
    calvin_note: "Early subtasks are already reliable, and the gain appears where drift has had time to compound: +4.2 at three completed subtasks, +5.0 at four and +3.4 at five, so the average completed length rises from 3.49 to 3.63."
  },

  scope: [
    "On suites that are close to saturation the entropy-scaled weight stays small and so does the gain: π0.5 and OpenVLA-OFT improve Spatial, Object and Goal by 0.4 to 1.2 points.",
    "On SIMPLER Google Robot, CogACT loses 2.5 points on Move Near (85.4 → 82.9) while gaining on the other three tasks.",
    "On SIMPLER WidowX, SpatialVLA drops from 100.0 to 95.8 on Put Eggplant in Yellow Basket; over the 96 paired evaluations AWARE wins seven configurations and loses five.",
    "The strength is set per task suite. A single α = 0.03 on every suite still raises the average over the base policies, strengths of α ≥ 0.06 degrade precision, and autoregressive models prefer small values (α ≤ 0.03 for OpenVLA, α = 0.01 for LLaVA-VLA)."
  ],

  analysis: {
    lede: "All ablations use OpenVLA-OFT on LIBERO. Removing any part of the rule costs success; replacing the entropy routing or the retrieved content with a plausible alternative recovers only part of the gain; and the method holds across a broad range of strengths.",
    ablation_note: "Removing the depth factor costs the most (98.1 → 97.0 on average). LIBERO-Object is the most sensitive suite and loses at least 1.4 points under every ablation.",
    controls_note: "Only the full rule, which routes the model's own observation tokens to the high-entropy elements, reaches +2.2. A fixed schedule, shuffled entropies, another episode's memory, matched noise, or damping the FFN without injecting anything recover +0.2 to +1.0.",
    cost_note: "The instruction weighting is computed once per policy call and the observation memory is cached; at each layer AWARE touches only the 56 action elements, adding 2.6 ms per environment step."
  },

  videos: {
    lede: "Paired rollouts from the evaluation runs behind Tables I–III and Figure 3: within an example both clips compare the base policy against AWARE on the same task, and the percentage under a clip is that policy's success rate on the whole suite or task."
  },

  figures: {
    footprint: {
      group: "Figure 1", title: "Empirical motivation for AWARE",
      caption: "Empirical motivation for AWARE (OpenVLA-OFT on LIBERO). (a) Observation subspace footprint diminishes across network depth, halving by layer 26 across all suites. (b) Action Entropy across all 56 action chunk elements: near steps settle early in middle layers while distant steps retain elevated predictive entropy until deep layers, forming a clear decision wavefront (white dashed isocontour) that motivates element-wise observation routing.",
      alt: "Two panels: observation share falling with layer for four LIBERO suites, and a heatmap of action entropy over 56 chunk elements and 32 layers.",
      files: { svg: "assets/img/fig_footprint.svg" }
    },
    aware_pipeline: {
      group: "Figure 2", title: "Overall framework of AWARE",
      caption: "Overall framework of AWARE. At every layer the language head reads each element of the predicted action representations and gives its Action Entropy. The observation features form a key-value memory that the element queries through its FFN input, and the retrieved features enter its FFN output at the next layer with weight w^{(ℓ+1)}_{t,n} = α · ℓ/(L−1) · H^{(ℓ)}_{t,n}. Nothing is switched on or off, and the memory is read at every layer.",
      alt: "Diagram of AWARE inside a VLA backbone: visual, text and proprioceptive tokens pass through the transformer layers; the language head gives per-element action entropy, which sets the weight of observation evidence retrieved from a key-value memory into each action element's FFN output.",
      files: { webp: "assets/img/aware_pipeline.webp" }
    },
    alpha: {
      group: "Figure 5", title: "Impact of the strength α",
      caption: "Impact of the strength α. Success rate per suite across the parameter grid; α = 0 denotes the untouched baseline policy, and outlined markers denote the optimal operating points for each suite (reported in Table I).",
      alt: "Success rate per LIBERO suite as a function of the strength alpha for OpenVLA-OFT and pi0.",
      files: { svg: "assets/img/fig_alpha.svg" }
    },
    real_world: {
      group: "Figure 3 (top)", title: "Real-world setup",
      caption: "Real-world experimental setup: the Franka Research 3 robot workstation (left) and the five evaluated manipulation tasks (right).",
      alt: "Photo of the Franka Research 3 workstation with a ZED 2i camera and 3D mouse, and two keyframes for each of the five tasks.",
      files: { webp: "assets/img/real_world_setup.webp" }
    }
  }
};
