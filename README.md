# CarRacing-v0 with Actor Critic Network + PPO

Solving CarRacing-v0 with actor critic network and proximal policy optimization.

### Preprocessing

Remove the base part that shows the score, and rescale to 96x96.


### Notes

- Reasonable results, albeit with a lot of jerky steering.

- Best result so far was on following:
    - Train by letting the network optimize speed, steering and braking for initial bit.
    - Later on, switch off the throttle, and set to constant. This allowed for good convergence.

- Modifying parameters during training helps fine tune.
    - Threshold for greenery death can be decayed
    - Penalty for jerky steering should be introduced later into training
    - Clipping parameter can be decayed.