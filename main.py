from comet_ml import Experiment
from time import sleep
import numpy as np
from agentFile import Agent
from environment import Env
from config import configure
from API_KEYS import api_key, project_name

configs, use_cuda,  device = configure()

## SET LOGGING
experiment = Experiment(project_name = project_name,  api_key = api_key)
experiment.log_parameters(configs.getParamsDict())
    

def getTrainTest( isTest = False, experiment = None,):
    if isTest:
        return experiment.test()
    return experiment.train()

if __name__ == "__main__":
    
    agent = Agent(configs.checkpoint, configs, device)
    env = Env(configs)
        
    with getTrainTest(configs.test, experiment):
        print('-------------BEGINNING EXPERIMENT--------------')
        for episodeIndex in range(configs.checkpoint, 100000):
            score = 0
            prevState = env.reset()
            for t in range(10000):
                action, a_logp = agent.select_action(prevState)
                
                curState, reward, dead, reasonForDeath = env.step(action, t, agent)
                
                if not configs.test:
                    agent.update((prevState, action, a_logp, reward, curState), episodeIndex)
                score += reward
                prevState = curState

                if dead:
                    print('--------------------')
                    print("Dead at score = ", round(score, 2), ' || Timesteps = ', t, ' || Reason = ', reasonForDeath)
                    break
            
            experiment.log_metric("scores", score , step= episodeIndex)

            print('Ep {}\tLast score: {:.2f}\n--------------------\n'.format(episodeIndex, score))
