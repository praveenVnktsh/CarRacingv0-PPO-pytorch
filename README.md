# CarRacing-v0 with Actor Critic Network + PPO

Solving CarRacing-v0 with actor critic network and proximal policy optimization.

### Preprocessing

Convert 96x96 into distances in front of the car!


### Notes

- Reasonable results, albeit with a lot of jerky steering.

- Best result so far was on following:
    - Train by letting the network optimize speed, steering and braking for initial bit.
    - Later on, switch off the throttle, and set to constant. This allowed for good convergence.

- Modifying parameters during training helps fine tune.
    - Threshold for greenery death can be decayed
    - Penalty for jerky steering should be introduced later into training
    - Clipping parameter can be decayed.


- A memory leak problem with the OpenAI Gym was resolved by following these steps:

> https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/envs/box2d/car_racing.py#L527
> 
> There is memory leaking in car_racing.py code.
> 
> pyglet.graphics.vertex_list() function makes the memory almost double,
> 
> And it can be fixed with `vl.delete()` under this line
> https://github.com/openai/gym/blob/38a1f630dc9815a567aaf299ae5844c8f8b9a6fa/gym/envs/box2d/car_racing.py#L530
