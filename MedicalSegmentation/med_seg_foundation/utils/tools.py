import numpy as np
import wandb

def get_class_rel_freqs(dataset):
    class_counts = np.zeros(dataset.num_classes)
    total_pixels = 0
    
    for idx in range(len(dataset)):
        tup = dataset[idx]
        mask = tup[1]
        
        class_counts += np.bincount(mask.flatten(), minlength=dataset.num_classes)  # Update class counts
        total_pixels += mask.flatten().size()[0]  # Update total pixels
        
    # Calculate relative frequencies
    relative_frequencies = class_counts / total_pixels
    assert relative_frequencies.size == dataset.num_classes
    return relative_frequencies

def log_class_rel_freqs(dataset, log_name_key):
    rel_freqs = get_class_rel_freqs(dataset)
    num_classes = rel_freqs.size
    
    for i in range(num_classes):
        metric_n = f'{log_name_key}_rel_freq_class{i}'
        wandb.define_metric(metric_n, summary="max")
        wandb.log({metric_n : rel_freqs[i]})