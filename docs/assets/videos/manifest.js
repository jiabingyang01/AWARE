/* Paired rollout clips, one list per benchmark. LIBERO: OpenVLA-OFT, base policy vs. AWARE at the Table I
   strength, same task and standard initial state. SIMPLER: SpatialVLA on WidowX, base vs. AWARE (alpha = 0.08),
   same task and episode configuration. CALVIN: LLaVA-VLA on ABC->D 5-subtask chains, base vs. AWARE (alpha = 0.01).
   Real: five VLA backbones on the five Franka Research 3 tabletop tasks (overhead ZED 2i camera), base vs. AWARE.
   Built by tools/build_videos.py. */
window.ROLLOUTS = {
 "LIBERO": [
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Spatial",
   "task": "task4",
   "title": "Bowl from the top drawer",
   "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
   "episode": 18,
   "episode_label": "initial state",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/spatial_task4_trial18_base.mp4",
     "poster": "assets/videos/libero/spatial_task4_trial18_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/spatial_task4_trial18_aware.mp4",
     "poster": "assets/videos/libero/spatial_task4_trial18_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Spatial",
   "task": "task5",
   "title": "Bowl on the ramekin",
   "instruction": "pick up the black bowl on the ramekin and place it on the plate",
   "episode": 24,
   "episode_label": "initial state",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/spatial_task5_trial24_base.mp4",
     "poster": "assets/videos/libero/spatial_task5_trial24_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/spatial_task5_trial24_aware.mp4",
     "poster": "assets/videos/libero/spatial_task5_trial24_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Object",
   "task": "task1",
   "title": "Cream cheese to the basket",
   "instruction": "pick up the cream cheese and place it in the basket",
   "episode": 27,
   "episode_label": "initial state",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/object_task1_trial27_base.mp4",
     "poster": "assets/videos/libero/object_task1_trial27_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/object_task1_trial27_aware.mp4",
     "poster": "assets/videos/libero/object_task1_trial27_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Object",
   "task": "task5",
   "title": "Tomato sauce to the basket",
   "instruction": "pick up the tomato sauce and place it in the basket",
   "episode": 37,
   "episode_label": "initial state",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/object_task5_trial37_base.mp4",
     "poster": "assets/videos/libero/object_task5_trial37_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/object_task5_trial37_aware.mp4",
     "poster": "assets/videos/libero/object_task5_trial37_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Goal",
   "task": "task0",
   "title": "Open the middle drawer",
   "instruction": "open the middle drawer of the cabinet",
   "episode": 30,
   "episode_label": "initial state",
   "alpha": 0.1,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/goal_task0_trial30_base.mp4",
     "poster": "assets/videos/libero/goal_task0_trial30_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/goal_task0_trial30_aware.mp4",
     "poster": "assets/videos/libero/goal_task0_trial30_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Goal",
   "task": "task9",
   "title": "Wine bottle on the rack",
   "instruction": "put the wine bottle on the rack",
   "episode": 11,
   "episode_label": "initial state",
   "alpha": 0.1,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/goal_task9_trial11_base.mp4",
     "poster": "assets/videos/libero/goal_task9_trial11_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/goal_task9_trial11_aware.mp4",
     "poster": "assets/videos/libero/goal_task9_trial11_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Long",
   "task": "task0",
   "title": "Soup and sauce in the basket",
   "instruction": "put both the alphabet soup and the tomato sauce in the basket",
   "episode": 16,
   "episode_label": "initial state",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/long_task0_trial16_base.mp4",
     "poster": "assets/videos/libero/long_task0_trial16_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/long_task0_trial16_aware.mp4",
     "poster": "assets/videos/libero/long_task0_trial16_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "suite": "Long",
   "task": "task6",
   "title": "Mug on the plate, pudding to its right",
   "instruction": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
   "episode": 10,
   "episode_label": "initial state",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/libero/long_task6_trial10_base.mp4",
     "poster": "assets/videos/libero/long_task6_trial10_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/libero/long_task6_trial10_aware.mp4",
     "poster": "assets/videos/libero/long_task6_trial10_aware.jpg",
     "success": true
    }
   ]
  }
 ],
 "SIMPLER": [
  {
   "backbone": "SpatialVLA",
   "column": "Put Carrot on Plate",
   "task": "PutCarrotOnPlate",
   "title": "Put carrot on plate",
   "instruction": "put carrot on plate",
   "episode": 11,
   "episode_label": "configuration",
   "alpha": 0.08,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/simpler/PutCarrotOnPlate_ep11_base.mp4",
     "poster": "assets/videos/simpler/PutCarrotOnPlate_ep11_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/simpler/PutCarrotOnPlate_ep11_aware.mp4",
     "poster": "assets/videos/simpler/PutCarrotOnPlate_ep11_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "SpatialVLA",
   "column": "Put Spoon on Towel",
   "task": "PutSpoonOnTableCloth",
   "title": "Put spoon on towel",
   "instruction": "put the spoon on the towel",
   "episode": 1,
   "episode_label": "configuration",
   "alpha": 0.08,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/simpler/PutSpoonOnTableCloth_ep1_base.mp4",
     "poster": "assets/videos/simpler/PutSpoonOnTableCloth_ep1_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/simpler/PutSpoonOnTableCloth_ep1_aware.mp4",
     "poster": "assets/videos/simpler/PutSpoonOnTableCloth_ep1_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "SpatialVLA",
   "column": "Stack Green Block on Yellow Block",
   "task": "StackGreenCubeOnYellowCube",
   "title": "Stack green block on yellow block",
   "instruction": "stack the green block on the yellow block",
   "episode": 9,
   "episode_label": "configuration",
   "alpha": 0.08,
   "chip": "Stack green block, configuration 9",
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep9_base.mp4",
     "poster": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep9_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep9_aware.mp4",
     "poster": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep9_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "SpatialVLA",
   "column": "Stack Green Block on Yellow Block",
   "task": "StackGreenCubeOnYellowCube",
   "title": "Stack green block on yellow block",
   "instruction": "stack the green block on the yellow block",
   "episode": 14,
   "episode_label": "configuration",
   "alpha": 0.08,
   "chip": "Stack green block, configuration 14",
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep14_base.mp4",
     "poster": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep14_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep14_aware.mp4",
     "poster": "assets/videos/simpler/StackGreenCubeOnYellowCube_ep14_aware.jpg",
     "success": true
    }
   ]
  }
 ],
 "CALVIN": [
  {
   "backbone": "LLaVA-VLA",
   "task": "seq402",
   "title": "Push pink block → open drawer → slide door → lift red block → place in drawer",
   "chip": "Chain #402 (0/5 → 5/5)",
   "instruction": "go push the pink block right → pull the handle to open the drawer → push the sliding door to the right side → grasp and lift the red block → store the grasped block in the drawer",
   "episode": 402,
   "episode_label": "chain",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/calvin/seq402_base.mp4",
     "poster": "assets/videos/calvin/seq402_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/calvin/seq402_aware.mp4",
     "poster": "assets/videos/calvin/seq402_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "seq201",
   "title": "Rotate pink block → slide door → turn on LED → close drawer → lift pink block",
   "chip": "Chain #201 (0/5 → 5/5)",
   "instruction": "take the pink block and rotate it to the right → push the sliding door to the right side → press the button to turn on the led light → push the handle to close the drawer → grasp and lift the pink block",
   "episode": 201,
   "episode_label": "chain",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/calvin/seq201_base.mp4",
     "poster": "assets/videos/calvin/seq201_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/calvin/seq201_aware.mp4",
     "poster": "assets/videos/calvin/seq201_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "seq232",
   "title": "Lift blue block → stack block → slide door → unstack block → push pink block",
   "chip": "Chain #232 (1/5 → 5/5)",
   "instruction": "lift the blue block from the sliding cabinet → stack the grasped block → push the sliding door to the left side → remove the stacked block → go push the pink block left",
   "episode": 232,
   "episode_label": "chain",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/calvin/seq232_base.mp4",
     "poster": "assets/videos/calvin/seq232_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/calvin/seq232_aware.mp4",
     "poster": "assets/videos/calvin/seq232_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "seq339",
   "title": "Lift blue block → place in slider → slide door → close drawer → lift red block",
   "chip": "Chain #339 (1/5 → 5/5)",
   "instruction": "grasp and lift the blue block → store the grasped block in the sliding cabinet → push the sliding door to the right side → push the handle to close the drawer → lift the red block from the sliding cabinet",
   "episode": 339,
   "episode_label": "chain",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/calvin/seq339_base.mp4",
     "poster": "assets/videos/calvin/seq339_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/calvin/seq339_aware.mp4",
     "poster": "assets/videos/calvin/seq339_aware.jpg",
     "success": true
    }
   ]
  }
 ],
 "Real": [
  {
   "backbone": "LLaVA-VLA",
   "task": "corn_to_pink_plate",
   "title": "Put Corn on Pink Plate",
   "instruction": "Put the corn on the pink plate",
   "kind": "short-horizon placement",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/llava/llava_corn_to_pink_plate_base.mp4",
     "poster": "assets/videos/realworld/llava/llava_corn_to_pink_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/llava/llava_corn_to_pink_plate_aware.mp4",
     "poster": "assets/videos/realworld/llava/llava_corn_to_pink_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "press_yellow_then_blue_button",
   "title": "Press Yellow Button then Blue",
   "instruction": "Press the yellow button, then press the blue button",
   "kind": "sequential button pressing",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/llava/llava_press_yellow_then_blue_button_base.mp4",
     "poster": "assets/videos/realworld/llava/llava_press_yellow_then_blue_button_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/llava/llava_press_yellow_then_blue_button_aware.mp4",
     "poster": "assets/videos/realworld/llava/llava_press_yellow_then_blue_button_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "box_cover_blue_block",
   "title": "Cover Blue Block with the Box",
   "instruction": "Cover the blue block with the box",
   "kind": "obstacle occlusion",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/llava/llava_box_cover_blue_block_base.mp4",
     "poster": "assets/videos/realworld/llava/llava_box_cover_blue_block_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/llava/llava_box_cover_blue_block_aware.mp4",
     "poster": "assets/videos/realworld/llava/llava_box_cover_blue_block_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "corn_to_upper_rack",
   "title": "Lift Corn onto Upper Rack Shelf",
   "instruction": "Put the corn on the upper rack",
   "kind": "multi-tier precision lifting",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/llava/llava_corn_to_upper_rack_base.mp4",
     "poster": "assets/videos/realworld/llava/llava_corn_to_upper_rack_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/llava/llava_corn_to_upper_rack_aware.mp4",
     "poster": "assets/videos/realworld/llava/llava_corn_to_upper_rack_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "LLaVA-VLA",
   "task": "tomato_from_turntable_to_white_plate",
   "title": "Move Tomato from Turntable to Plate",
   "instruction": "Pick up the tomato from the rotating turntable and place it in the white plate",
   "kind": "dynamic moving-target acquisition",
   "alpha": 0.01,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/llava/llava_tomato_from_turntable_to_white_plate_base.mp4",
     "poster": "assets/videos/realworld/llava/llava_tomato_from_turntable_to_white_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/llava/llava_tomato_from_turntable_to_white_plate_aware.mp4",
     "poster": "assets/videos/realworld/llava/llava_tomato_from_turntable_to_white_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "task": "corn_to_pink_plate",
   "title": "Put Corn on Pink Plate",
   "instruction": "Put the corn on the pink plate",
   "kind": "short-horizon placement",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/oft/oft_corn_to_pink_plate_base.mp4",
     "poster": "assets/videos/realworld/oft/oft_corn_to_pink_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/oft/oft_corn_to_pink_plate_aware.mp4",
     "poster": "assets/videos/realworld/oft/oft_corn_to_pink_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "task": "press_yellow_then_blue_button",
   "title": "Press Yellow Button then Blue",
   "instruction": "Press the yellow button, then press the blue button",
   "kind": "sequential button pressing",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/oft/oft_press_yellow_then_blue_button_base.mp4",
     "poster": "assets/videos/realworld/oft/oft_press_yellow_then_blue_button_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/oft/oft_press_yellow_then_blue_button_aware.mp4",
     "poster": "assets/videos/realworld/oft/oft_press_yellow_then_blue_button_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "task": "box_cover_blue_block",
   "title": "Cover Blue Block with the Box",
   "instruction": "Cover the blue block with the box",
   "kind": "obstacle occlusion",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/oft/oft_box_cover_blue_block_base.mp4",
     "poster": "assets/videos/realworld/oft/oft_box_cover_blue_block_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/oft/oft_box_cover_blue_block_aware.mp4",
     "poster": "assets/videos/realworld/oft/oft_box_cover_blue_block_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "task": "corn_to_upper_rack",
   "title": "Lift Corn onto Upper Rack Shelf",
   "instruction": "Put the corn on the upper rack",
   "kind": "multi-tier precision lifting",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/oft/oft_corn_to_upper_rack_base.mp4",
     "poster": "assets/videos/realworld/oft/oft_corn_to_upper_rack_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/oft/oft_corn_to_upper_rack_aware.mp4",
     "poster": "assets/videos/realworld/oft/oft_corn_to_upper_rack_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "OpenVLA-OFT",
   "task": "tomato_from_turntable_to_white_plate",
   "title": "Move Tomato from Turntable to Plate",
   "instruction": "Pick up the tomato from the rotating turntable and place it in the white plate",
   "kind": "dynamic moving-target acquisition",
   "alpha": 0.04,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/oft/oft_tomato_from_turntable_to_white_plate_base.mp4",
     "poster": "assets/videos/realworld/oft/oft_tomato_from_turntable_to_white_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/oft/oft_tomato_from_turntable_to_white_plate_aware.mp4",
     "poster": "assets/videos/realworld/oft/oft_tomato_from_turntable_to_white_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "CogACT",
   "task": "corn_to_pink_plate",
   "title": "Put Corn on Pink Plate",
   "instruction": "Put the corn on the pink plate",
   "kind": "short-horizon placement",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/cogact/cogact_corn_to_pink_plate_base.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_corn_to_pink_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/cogact/cogact_corn_to_pink_plate_aware.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_corn_to_pink_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "CogACT",
   "task": "press_yellow_then_blue_button",
   "title": "Press Yellow Button then Blue",
   "instruction": "Press the yellow button, then press the blue button",
   "kind": "sequential button pressing",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/cogact/cogact_press_yellow_then_blue_button_base.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_press_yellow_then_blue_button_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/cogact/cogact_press_yellow_then_blue_button_aware.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_press_yellow_then_blue_button_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "CogACT",
   "task": "box_cover_blue_block",
   "title": "Cover Blue Block with the Box",
   "instruction": "Cover the blue block with the box",
   "kind": "obstacle occlusion",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/cogact/cogact_box_cover_blue_block_base.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_box_cover_blue_block_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/cogact/cogact_box_cover_blue_block_aware.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_box_cover_blue_block_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "CogACT",
   "task": "corn_to_upper_rack",
   "title": "Lift Corn onto Upper Rack Shelf",
   "instruction": "Put the corn on the upper rack",
   "kind": "multi-tier precision lifting",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/cogact/cogact_corn_to_upper_rack_base.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_corn_to_upper_rack_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/cogact/cogact_corn_to_upper_rack_aware.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_corn_to_upper_rack_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "CogACT",
   "task": "tomato_from_turntable_to_white_plate",
   "title": "Move Tomato from Turntable to Plate",
   "instruction": "Pick up the tomato from the rotating turntable and place it in the white plate",
   "kind": "dynamic moving-target acquisition",
   "alpha": 0.05,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/cogact/cogact_tomato_from_turntable_to_white_plate_base.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_tomato_from_turntable_to_white_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/cogact/cogact_tomato_from_turntable_to_white_plate_aware.mp4",
     "poster": "assets/videos/realworld/cogact/cogact_tomato_from_turntable_to_white_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0",
   "task": "corn_to_pink_plate",
   "title": "Put Corn on Pink Plate",
   "instruction": "Put the corn on the pink plate",
   "kind": "short-horizon placement",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi0/pi0_corn_to_pink_plate_base.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_corn_to_pink_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi0/pi0_corn_to_pink_plate_aware.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_corn_to_pink_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0",
   "task": "press_yellow_then_blue_button",
   "title": "Press Yellow Button then Blue",
   "instruction": "Press the yellow button, then press the blue button",
   "kind": "sequential button pressing",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi0/pi0_press_yellow_then_blue_button_base.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_press_yellow_then_blue_button_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi0/pi0_press_yellow_then_blue_button_aware.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_press_yellow_then_blue_button_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0",
   "task": "box_cover_blue_block",
   "title": "Cover Blue Block with the Box",
   "instruction": "Cover the blue block with the box",
   "kind": "obstacle occlusion",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi0/pi0_box_cover_blue_block_base.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_box_cover_blue_block_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi0/pi0_box_cover_blue_block_aware.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_box_cover_blue_block_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0",
   "task": "corn_to_upper_rack",
   "title": "Lift Corn onto Upper Rack Shelf",
   "instruction": "Put the corn on the upper rack",
   "kind": "multi-tier precision lifting",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi0/pi0_corn_to_upper_rack_base.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_corn_to_upper_rack_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi0/pi0_corn_to_upper_rack_aware.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_corn_to_upper_rack_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0",
   "task": "tomato_from_turntable_to_white_plate",
   "title": "Move Tomato from Turntable to Plate",
   "instruction": "Pick up the tomato from the rotating turntable and place it in the white plate",
   "kind": "dynamic moving-target acquisition",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi0/pi0_tomato_from_turntable_to_white_plate_base.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_tomato_from_turntable_to_white_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi0/pi0_tomato_from_turntable_to_white_plate_aware.mp4",
     "poster": "assets/videos/realworld/pi0/pi0_tomato_from_turntable_to_white_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0.5",
   "task": "corn_to_pink_plate",
   "title": "Put Corn on Pink Plate",
   "instruction": "Put the corn on the pink plate",
   "kind": "short-horizon placement",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi05/pi05_corn_to_pink_plate_base.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_corn_to_pink_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi05/pi05_corn_to_pink_plate_aware.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_corn_to_pink_plate_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0.5",
   "task": "press_yellow_then_blue_button",
   "title": "Press Yellow Button then Blue",
   "instruction": "Press the yellow button, then press the blue button",
   "kind": "sequential button pressing",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi05/pi05_press_yellow_then_blue_button_base.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_press_yellow_then_blue_button_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi05/pi05_press_yellow_then_blue_button_aware.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_press_yellow_then_blue_button_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0.5",
   "task": "box_cover_blue_block",
   "title": "Cover Blue Block with the Box",
   "instruction": "Cover the blue block with the box",
   "kind": "obstacle occlusion",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi05/pi05_box_cover_blue_block_base.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_box_cover_blue_block_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi05/pi05_box_cover_blue_block_aware.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_box_cover_blue_block_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0.5",
   "task": "corn_to_upper_rack",
   "title": "Lift Corn onto Upper Rack Shelf",
   "instruction": "Put the corn on the upper rack",
   "kind": "multi-tier precision lifting",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi05/pi05_corn_to_upper_rack_base.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_corn_to_upper_rack_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi05/pi05_corn_to_upper_rack_aware.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_corn_to_upper_rack_aware.jpg",
     "success": true
    }
   ]
  },
  {
   "backbone": "π0.5",
   "task": "tomato_from_turntable_to_white_plate",
   "title": "Move Tomato from Turntable to Plate",
   "instruction": "Pick up the tomato from the rotating turntable and place it in the white plate",
   "kind": "dynamic moving-target acquisition",
   "alpha": 0.03,
   "clips": [
    {
     "rule": "Base",
     "file": "assets/videos/realworld/pi05/pi05_tomato_from_turntable_to_white_plate_base.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_tomato_from_turntable_to_white_plate_base.jpg",
     "success": false
    },
    {
     "rule": "AWARE",
     "file": "assets/videos/realworld/pi05/pi05_tomato_from_turntable_to_white_plate_aware.mp4",
     "poster": "assets/videos/realworld/pi05/pi05_tomato_from_turntable_to_white_plate_aware.jpg",
     "success": true
    }
   ]
  }
 ]
};
