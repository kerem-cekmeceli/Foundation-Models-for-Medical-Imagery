import math
from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data import DistributedSampler    


""" 
    THIS MODULE IS NOT ACTIVE AND HAS BUGS ! UNEVEN NB OF SAMPLES FOR EACH VOLUME IS PROBLEMATIC
"""

class VolDistributedSampler(DistributedSampler):
    def __init__(self, dataset: Dataset,  vol_depth:int, num_replicas: Optional[int] = None, 
                 rank: Optional[int] = None, shuffle: bool = True, 
                 seed: int = 0, drop_last: bool = False) -> None:
        
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        
        assert vol_depth>0
        self.vol_depth = vol_depth
        
        assert len(self.dataset)> 0
        
        assert len(self.dataset)%self.vol_depth == 0, "Dataset can not contain a partial volume"
        self.total_nb_vols_dataset = len(self.dataset)//self.vol_depth
        assert self.total_nb_vols_dataset > 0
        
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and self.total_nb_vols_dataset % self.num_replicas != 0 and \
            self.total_nb_vols_dataset>self.num_replicas:  
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_vols = math.floor(self.total_nb_vols_dataset / self.num_replicas)

        else:
            # Num volumes per rank
            self.num_vols = math.ceil(self.total_nb_vols_dataset / self.num_replicas)
            
        assert self.num_vols>0, "Each GPU must get at least 1 volume"
                        
        # Num samples per rank
        self.num_samples = self.num_vols * self.vol_depth
        
        # Total number of volumes (inc replication)
        self.total_nb_vols = self.num_vols * self.num_replicas
        self.total_size = self.total_nb_vols * self.vol_depth
        
    def __iter__(self):
        if self.shuffle:
            # # deterministically shuffle based on epoch and seed
            # g = torch.Generator()
            # g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            ValueError('Not Implemented yet') # shuffle assigned volumes and slices in the vol but keep complete volumes for all GPUs
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            
            if padding_size>0:
                assert padding_size%self.vol_depth == 0
            
            if padding_size <= len(indices):
                indices += indices[:padding_size]  # append from the first volumes 
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        start_idx = self.rank*self.num_samples
        end_idx = start_idx + self.num_samples
        indices = indices[start_idx:end_idx:1]
        assert len(indices) == self.num_samples

        return iter(indices) 
  
  