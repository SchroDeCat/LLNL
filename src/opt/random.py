
'''
Implementation for experiments on LLNL simulation env.
'''
import numpy as np
import matplotlib.pyplot as plt

class RandomOpt:
    def __init__(self, y, name:str, ):
        self.y = y
        self.max = y.squeeze().max()
        self.name = name
        
    
    def opt(self, horizon:int, repeat:int=10):
        self.horizon = horizon
        self.repeat = repeat
        self.regret = np.zeros([self.repeat, self.horizon])
        for i in range(self.repeat):
            self._selection = np.random.choice(self.y.squeeze().size(0), horizon, replace=True)
            self._ys = self.y.squeeze()[self._selection]
            # print(f"self._ys {self._ys.shape} self.y {self.y.shape} ")
            self.regret[i] = np.minimum.accumulate(self.max - self._ys)
    

    def store_results(self, dir:str):
        np.save(f"{self.__file_name(dir)}.npy", self.regret)
        pass

    def plot_results(self, dir:str):
        plt.plot(self.regret.mean(axis=0), label=self.name)
        plt.xlabel("iter")
        plt.title("simple_regret")
        plt.savefig(f"{self.__file_name(dir)}.png")
        pass

    def __file_name(self, dir):
        return f"{dir}/RandomOpt-{self.name}-h{self.horizon}-r{self.repeat}"