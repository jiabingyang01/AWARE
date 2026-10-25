/* Numbers of the paper "AWARE: Action-Entropy-Weighted and Attention-Guided Retrieval of Observation Evidence for
   Vision-Language-Action Models", copied from its tables. Success rates are in percent. */
window.PAPER_DATA = {
  setup: {
    backbones: [
      { name: "LLaVA-VLA", size: "0.5B", family: "single-system", benchmarks: ["CALVIN"] },
      { name: "SpatialVLA", size: "4B", family: "single-system", benchmarks: ["SIMPLER"] },
      { name: "OpenVLA", size: "7B", family: "single-system", benchmarks: ["LIBERO"] },
      { name: "OpenVLA-OFT", size: "7B", family: "single-system", benchmarks: ["LIBERO"] },
      { name: "π0", size: "3B", family: "dual-system", benchmarks: ["LIBERO"] },
      { name: "π0.5", size: "3B", family: "dual-system", benchmarks: ["LIBERO"] },
      { name: "CogACT", size: "7B", family: "dual-system", benchmarks: ["SIMPLER"] }
    ],
    alpha_range: [0.01, 0.10],
    gpu: "NVIDIA A100",
    LIBERO: { suites: ["Spatial", "Object", "Goal", "Long"], tasks_per_suite: 10, episodes_per_suite: 500 },
    SIMPLER: { protocol: "Visual Matching", platforms: ["Google Robot", "WidowX"] },
    CALVIN: { split: "ABC→D", chains: 500 },
    OFT: { H: 8, D_a: 7, N_a: 56 },
    K: 256,
    real_world: {
      robot: "Franka Research 3 with a parallel-jaw gripper",
      camera: "static overhead ZED 2i",
      teleoperation: "3Dconnexion 3D mouse",
      demos_per_task: 50,
      backbones: ["LLaVA-VLA", "OpenVLA-OFT", "CogACT", "π0", "π0.5"],
      alpha: { "LLaVA-VLA": 0.01, "OpenVLA-OFT": 0.04, "CogACT": 0.05, "π0": 0.03, "π0.5": 0.03 },
      tasks: [
        { key: "corn_to_pink_plate", name: "Put Corn on Pink Plate", instruction: "Put the corn on the pink plate", kind: "short-horizon placement" },
        { key: "press_yellow_then_blue_button", name: "Press Yellow Button then Blue", instruction: "Press the yellow button, then press the blue button", kind: "sequential button pressing" },
        { key: "box_cover_blue_block", name: "Cover Blue Block with the Box", instruction: "Cover the blue block with the box", kind: "obstacle occlusion" },
        { key: "corn_to_upper_rack", name: "Lift Corn onto Upper Rack Shelf", instruction: "Put the corn on the upper rack", kind: "multi-tier precision lifting" },
        { key: "tomato_from_turntable_to_white_plate", name: "Move Tomato from Turntable to Plate", instruction: "Pick up the tomato from the rotating turntable and place it in the white plate", kind: "dynamic moving-target acquisition", moving: true }
      ]
    }
  },

  /* Table I */
  libero: {
    label: "Table I",
    caption: "Performance comparison on the LIBERO benchmark. All baselines are evaluated under identical checkpoints and evaluation pipelines. Bold denotes the better of a pair.",
    columns: ["Spatial", "Object", "Goal", "Long", "Average"],
    models: [
      { name: "OpenVLA", venue: "CoRL'24", alpha: { Spatial: 0.03, Object: 0.02, Goal: 0.01, Long: 0.01 },
        base: [82.8, 85.6, 78.4, 49.6, 74.1], aware: [86.4, 87.0, 79.0, 57.2, 77.4], delta: [3.6, 1.4, 0.6, 7.6, 3.3] },
      { name: "π0", venue: "RSS'25", alpha: { Spatial: 0.03, Object: 0.04, Goal: 0.05, Long: 0.03 },
        base: [96.4, 97.2, 93.8, 81.2, 92.2], aware: [98.2, 98.6, 94.4, 84.0, 93.8], delta: [1.8, 1.4, 0.6, 2.8, 1.6] },
      { name: "π0.5", venue: "CoRL'25", alpha: { Spatial: 0.01, Object: 0.08, Goal: 0.01, Long: 0.07 },
        base: [98.2, 98.6, 97.6, 92.6, 96.8], aware: [99.2, 99.6, 98.4, 94.0, 97.8], delta: [1.0, 1.0, 0.8, 1.4, 1.0] },
      { name: "OpenVLA-OFT", venue: "RSS'25", alpha: { Spatial: 0.04, Object: 0.05, Goal: 0.10, Long: 0.03 },
        base: [98.0, 98.2, 97.6, 93.8, 96.9], aware: [99.2, 98.6, 98.6, 96.0, 98.1], delta: [1.2, 0.4, 1.0, 2.2, 1.2] }
    ]
  },

  /* Table II */
  simpler: {
    label: "Table II",
    caption: "Performance comparison on the SIMPLER benchmark (Visual Matching). Evaluated with official released checkpoints under identical environments and pipelines. Bold denotes the better of a pair.",
    blocks: [
      { robot: "Google Robot", model: "CogACT", venue: "arXiv'24", alpha: 0.09,
        columns: ["Pick Coke Can", "Move Near", "Open/Close Drawer", "Open Top Drawer and Place Apple", "Average"],
        base: [90.0, 85.4, 70.4, 49.1, 73.7], aware: [92.3, 82.9, 75.0, 57.4, 76.9], delta: [2.3, -2.5, 4.6, 8.3, 3.2] },
      { robot: "WidowX Robot", model: "SpatialVLA", venue: "RSS'25", alpha: 0.08,
        columns: ["Put Spoon on Towel", "Put Carrot on Plate", "Stack Green Block on Yellow Block", "Put Eggplant in Yellow Basket", "Average"],
        base: [16.7, 25.0, 29.2, 100.0, 42.7], aware: [20.8, 29.2, 33.3, 95.8, 44.8], delta: [4.2, 4.2, 4.2, -4.2, 2.1] }
    ],
    widowx_pairs: { evaluations: 96, wins: 7, losses: 5 }
  },

  /* Table III */
  calvin: {
    label: "Table III",
    caption: "Performance comparison on the CALVIN ABC→D benchmark, in success rate per chain length and average completed length. Evaluated with the released checkpoint; a single strength α = 0.01 is used across the entire benchmark.",
    model: "LLaVA-VLA", venue: "ICRA'26", alpha: 0.01,
    columns: ["1/5", "2/5", "3/5", "4/5", "5/5"],
    base: [94.0, 82.4, 68.2, 57.8, 47.0], aware: [94.4, 82.8, 72.4, 62.8, 50.4], delta: [0.4, 0.4, 4.2, 5.0, 3.4],
    avg_len: { base: 3.49, aware: 3.63, delta: 0.14, aware_exact: 3.628 }
  },

  /* Figure 3 (bottom) */
  real_world: {
    label: "Figure 3",
    trials_per_cell: 40,
    columns: ["corn_to_pink_plate", "press_yellow_then_blue_button", "box_cover_blue_block", "corn_to_upper_rack", "tomato_from_turntable_to_white_plate"],
    models: [
      { name: "LLaVA-VLA", alpha: 0.01, base: [75.0, 57.5, 65.0, 52.5, 55.0], aware: [82.5, 72.5, 77.5, 62.5, 50.0], delta: [7.5, 15.0, 12.5, 10.0, -5.0] },
      { name: "OpenVLA-OFT", alpha: 0.04, base: [87.5, 75.0, 72.5, 70.0, 67.5], aware: [92.5, 72.5, 90.0, 82.5, 65.0], delta: [5.0, -2.5, 17.5, 12.5, -2.5] },
      { name: "CogACT", alpha: 0.05, base: [82.5, 82.5, 80.0, 67.5, 70.0], aware: [90.0, 92.5, 80.0, 67.5, 77.5], delta: [7.5, 10.0, 0.0, 0.0, 7.5] },
      { name: "π0", alpha: 0.03, base: [90.0, 75.0, 85.0, 77.5, 67.5], aware: [90.0, 85.0, 90.0, 75.0, 77.5], delta: [0.0, 10.0, 5.0, -2.5, 10.0] },
      { name: "π0.5", alpha: 0.03, base: [92.5, 90.0, 87.5, 82.5, 80.0], aware: [95.0, 90.0, 92.5, 87.5, 87.5], delta: [2.5, 0.0, 5.0, 5.0, 7.5] }
    ]
  },

  /* Table IV */
  ablation: {
    label: "Table IV",
    caption: "Component analysis on the LIBERO benchmark with OpenVLA-OFT using the baseline strengths from Table I. Ablating individual components isolates the contribution of each design choice while holding the global intervention scale α fixed.",
    columns: ["Spatial", "Object", "Goal", "Long", "Average"],
    rows: [
      { key: "base", name: "Base policy", per_element: false, depth: false, instruction: false, values: [98.0, 98.2, 97.6, 93.8, 96.9] },
      { key: "no_element", name: "Uniform element weights", per_element: false, depth: true, instruction: true, values: [98.8, 97.0, 98.4, 95.2, 97.4] },
      { key: "no_depth", name: "No depth factor", per_element: true, depth: false, instruction: true, values: [98.2, 97.2, 97.8, 94.8, 97.0] },
      { key: "no_instruction", name: "No instruction guidance", per_element: true, depth: true, instruction: false, values: [98.6, 96.6, 98.4, 95.2, 97.2] },
      { key: "full", name: "AWARE (full rule)", per_element: true, depth: true, instruction: true, values: [99.2, 98.6, 98.6, 96.0, 98.1] }
    ]
  },

  /* Table V */
  controls: {
    label: "Table V",
    caption: "Diagnostic variants on OpenVLA-OFT (LIBERO-Long) testing alternative explanations across 500 evaluation episodes.",
    base: 93.8, aware: 96.0, aware_delta: 2.2,
    groups: [
      { aspect: "Uncertainty routing", rows: [
        { name: "Linear chunk schedule (w ∝ k/H)", value: 94.0, delta: 0.2 },
        { name: "Permuted entropy weights", value: 94.8, delta: 1.0 }
      ] },
      { aspect: "Injected content", rows: [
        { name: "Unrelated episode memory", value: 94.8, delta: 1.0 },
        { name: "Matched Gaussian noise", value: 94.6, delta: 0.8 },
        { name: "FFN damping w/o injection", value: 94.0, delta: 0.2 }
      ] }
    ]
  },

  /* Table VI */
  overhead: {
    label: "Table VI",
    caption: "Inference latency and throughput on OpenVLA-OFT (LIBERO-Long) evaluated on a single NVIDIA A100 GPU. Because the policy predicts action chunks of horizon H = 8, per-step latency reflects the effective wait time per environment interaction step.",
    rows: [
      { key: "throughput", name: "Throughput", unit: "Hz", better: "higher", base: 82.9, aware: 68.1, delta_pct: -17.9 },
      { key: "per_call", name: "Per call", unit: "ms", better: "lower", base: 96.5, aware: 117.5, delta_pct: 21.8 },
      { key: "per_step", name: "Per env. step", unit: "ms", better: "lower", base: 29.4, aware: 32.0, delta_pct: 8.9 }
    ],
    added_ms_per_step: 2.6
  },

  /* in-text numbers */
  text_numbers: {
    footprint_half_layer: 26,
    entropy_gap: 0.25,
    alpha_shared_optimum_long: 0.03,
    alpha_gain_long: { "OpenVLA-OFT": 2.2, "π0": 2.8 },
    peak_spatial: 99.2
  }
};
