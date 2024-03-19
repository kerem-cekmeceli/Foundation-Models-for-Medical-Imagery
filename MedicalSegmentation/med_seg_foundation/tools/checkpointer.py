import torch
import torch.nn as nn

from pathlib import Path

from queue import PriorityQueue
from copy import deepcopy

from typing import Union, List


class PriorityQueueFixedSz:
    def __init__(self, sz, keep_min=True):
        self.sz = sz
        self.minimize = keep_min
        self.pq = PriorityQueue(maxsize=sz)
        
    def update(self, score, data):            
        queue_score = -score if self.minimize else score
        new_el = (queue_score, data)
        
        if self.pq.full():
            old_worst = self.pq.get()
            if old_worst[0]<=new_el[0]:
                self.pq.put(new_el)
            else:
                self.pq.put(old_worst)
        else:
            self.pq.put((queue_score, data))

    def get_elems(self):
        els = []
        while not self.pq.empty():
            (queue_score, data) = self.pq.get()
            score = -queue_score if self.minimize else queue_score
            els.append((score, data))
        els.reverse()  # Best one first
        return els


class Checkpointer():
    def get_compliant_attr(attr, ref):
        if not isinstance(attr, list):
            attr = [attr]*len(ref)
        else:
            assert len(attr) == len(ref), 'shape mismatch'  
        return attr
    
    def __init__(self, 
                 monitor: Union[str, List[str]],
                 save_pth: Union[Path, str],
                 n_best: Union[int, List[int]]=1,
                 minimize: Union[bool, List[bool]]=True,
                 name_prefix:str='') -> None:
        
        if isinstance(monitor, str):
            assert isinstance(n_best, int) and isinstance(minimize, bool)
            monitor = [monitor]
            n_best = [n_best]
            minimize = [minimize]
        else:
            minimize = Checkpointer.get_compliant_attr(attr=minimize, ref=monitor)
            n_best = Checkpointer.get_compliant_attr(attr=n_best, ref=monitor)
            
        self.n_best = n_best
        self.minimize = minimize
        self.name_prefix = name_prefix
        self.monitor=monitor
        self.save_pth = save_pth
        
        if not isinstance(save_pth, Path):
            self.save_pth = Path(save_pth)
            self.save_pth.mkdir(parents=True, exist_ok=True)
        self.save_pth = str(self.save_pth)
        
        self.best = {}
        for i, m in enumerate(monitor):
            self.best[m] = PriorityQueueFixedSz(sz=n_best[i], keep_min=minimize[i])
                
    def update(self, 
               model:nn.Module,
               metrics:dict,
               epoch:int,
               opt:torch.optim.Optimizer=None,
               scheduler:torch.optim.lr_scheduler.LRScheduler=None):
        
        for m in self.monitor:
            assert m in metrics.keys(), \
                "monitored quantity: {m} is not found in the given metrics dict" 
            data = [epoch, model.state_dict()]
            
            if opt is not None:
                data.append(opt.state_dict())
                
            if scheduler is not None:
                data.append(scheduler.state_dict())
            
            self.best[m].update(score=metrics[m], data=data)
                   
    def save(self):
        for m in self.monitor:
            bests_m = self.best[m].get_elems()
            for i, (score, data) in enumerate(bests_m):
                epoch = data[0]
                state_dict = data[1]
                f_name = self.name_prefix + f'_{m}_top{i+1}_epoch{epoch}.pth'
                print(f'{m} top{i+1}: {round(score, 4)}, saved as: {f_name}, to {self.save_pth}')
                
                save_dict = dict(epoch = epoch,
                                 model_state_dict = state_dict)
                if len(data)>2:
                    save_dict['opt_state_dict'] = data[2]
                if len(data)>3:
                    save_dict['scheduler_state_dict'] = data[3]
                torch.save(save_dict, self.save_pth+'/'+f_name)
        