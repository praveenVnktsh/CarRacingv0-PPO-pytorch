from comet_ml import Experiment
import gc
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import configure

args, use_cuda,  device = configure()

## SET LOGGING
experiment = Experiment(project_name="CarRacing", api_key='P4Y69RtjtY1e0R20FCgvxtbi0' )
hyper_params = {
    "gamma": args.gamma,
    "action-repeat": args.action_repeat,
    "img-stack":args.img_stack,
    "seed": args.seed,
    "clip_param" : args.clip_param,
    "ppo_epoch" : args.ppo_epoch,
    "buffer_capacity" : args.buffer_capacity,
    "batch_size" : args.batch_size,
    "deathThreshold": args.deathThreshold,
    "saveLocation": args.saveLocation
}
experiment.log_parameters(hyper_params)

if __name__ == "__main__":
    
    checkpoint = 320
    with experiment.train():
        agent = Agent(checkpoint, args, device)
        env = Env(args)
        prevState = env.reset()
        for episodeIndex in range(checkpoint, 100000):
            score = 0
            prevState = env.reset()
            for t in range(10000):
                if t%200 - 1 == 0:
                    gc.collect()
                action, a_logp = agent.select_action(prevState)
                curState, reward, done, reason = env.step(action* np.array([-2., 0.0, 0.5]) + np.array([1., 0.5, 0.]), t)
                env.render()

                agent.update((prevState, action, a_logp, reward, curState), episodeIndex)

                score += reward
                prevState = curState

                if done:
                    print('--------------------')
                    print("Dead at score = ", round(score, 2), ' || Timesteps = ', t, ' || Reason = ', reason)
                    break
            gc.collect()
            experiment.log_metric("scores", score , step=episodeIndex)

            print('Ep {}\tLast score: {:.2f}\n--------------------\n'.format(episodeIndex, score))
