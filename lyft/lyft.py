#!/usr/bin/env python3

# map category ID to local presentation from 1 on.
# only process categories within this map
LABEL_MAP = {10: 1      # cars
            # 4: 2      # pedestrians, add extra like this
                        # currently supports only 1 category
            }


CATEGORIES = ['none', 'car']

IGNORE_SELF_CAR = True

SELF_CAR_H = 490
