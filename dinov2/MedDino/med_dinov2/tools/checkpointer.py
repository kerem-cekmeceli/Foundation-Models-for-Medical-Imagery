import torch
import torch.nn as nn

from pathlib import Path

from queue import PriorityQueue
from copy import deepcopy

from typing import Union

class Checkpointer():
    def __init__(self, 
                 monitor: str,
                 save_pth: Union[Path, str],
                 n_best: int=1,
                 minimize: bool=True,
                 name_prefix:str='') -> None:
        
        self.n_best = n_best
        self.minimize = minimize
        self.name_prefix = name_prefix
        self.monitor=monitor
        self.save_pth = save_pth
        
        if not isinstance(save_pth, Path):
            self.save_pth = Path(save_pth)
            self.save_pth.mkdir(parents=True, exist_ok=True)
        self.save_pth = str(self.save_pth)
        
        self.best = PriorityQueue(maxsize=3)
                
    def update(self, 
               model:nn.Module,
               metric,
               epoch:int):
        
        if self.minimize:
            score = metric
        else:
            score = -1*deepcopy(metric)
            
        self.best.put((score, (epoch, model.state_dict())))
    
    def save(self):
        i=1
        while not self.best.empty():
            (val, (epoch, model_state_dict)) = self.best.get() 
            f_name = self.name_prefix + f'_best{i}_epoch{epoch}.pth'
            print(f'Best {i}: {val}, saved as: {f_name}, to {self.save_pth}')
            torch.save(model_state_dict, self.save_pth+'/'+f_name)
            assert i<=self.n_best
            i += 1
        
        
        