

class Agent():

    def __init__(self, iterable=(), **kwargs):
        super().__init__(iterable, **kwargs)

        self.episodeCount = 0

        self.steps = 0
        
    def playEpisode(self):
        pass