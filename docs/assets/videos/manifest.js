/* Paired rollout clips, one list per benchmark. LIBERO: OpenVLA-OFT, base policy vs. AWARE at the Table I
   strength, same task and standard initial state. SIMPLER: SpatialVLA on WidowX, base vs. AWARE (alpha = 0.08),
   same task and episode configuration. Built by tools/build_videos.py. */
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
 ]
};
