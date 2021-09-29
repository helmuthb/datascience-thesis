# Names of classes used in object detection
det_classes = ["background", "buffer-stop", "crossing", "switch-indicator",
               "switch-left", "switch-right", "switch-static",
               "switch-unknown", "track-signal-back", "track-signal-front",
               "track-sign-front"]

# Names of classes used in segmentation
seg_classes = ["road", "sidewalk", "construction", "tram-track",
               "fence", "pole", "traffic-light", "traffic-sign",
               "vegetation", "terrain", "sky", "human", "rail-track",
               "car", "truck", "trackbed", "on-rails", "rail-raised",
               "rail-embedded"]

# subset of interest: object detection
det_subset = ["background", "track-signal-back", "track-signal-front",
              "track-sign-front"]

# subset of interest: segmentation
seg_subset = ["road", "rail-raised", "rail-embedded"]
