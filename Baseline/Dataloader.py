import torch
from torch.nn.utils.rnn import pad_sequence
from Indexer import Indexer
from typing import Dict, Any, List

class Collator:
    def __init__(self, indexer: Indexer) -> None:
        self._indexer = indexer
    
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        samples = {
            key: [sample[key] for sample in samples] for key in samples[0]
        }
        
        for key in 'start', 'end':
            if samples[key][0] is None:
                samples[key] = None
            else:
                samples[key] = torch.tensor(samples[key], dtype=torch.long)
        for key in 'input_ids', 'attention_mask', 'token_type_ids':
            samples[key] = pad_sequence(
                [torch.tensor(sample, dtype=torch.long) for sample in samples[key]],
                batch_first=True, padding_value=self._indexer.pad_id
            )
            
        return samples