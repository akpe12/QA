#%%
from typing import Sequence, List, Dict, Any
from collections import Counter
from itertools import chain
from tqdm import tqdm
from Tokenize import TokenizedKoMRC

class Indexer:
    def __init__(self,
                 id2token: List[str],
                 max_length: int=1024,
                 pad: str='<pad>',
                 unk: str='<unk>',
                 cls: str='<cls>',
                 sep: str='<sep>') -> None:
        self.pad = pad
        self.unk = unk
        self.cls = cls
        self.sep = sep
        self.special_tokens = [pad, unk, cls, sep]
        self.max_length = max_length
        self.id2token = self.special_tokens + id2token
        self.token2id = {token : token_id for token_id, token in enumerate(self.id2token)}
        
    @property
    def vocab_size(self):
        return len(self.id2token)
    
    @property
    def pad_id(self):
        return self.token2id[self.pad]
    
    @property
    def unk_id(self):
        return self.token2id[self.unk]
    
    @property
    def cls_id(self):
        return self.token2id[self.cls]
    
    @property
    def sep_id(self):
        return self.token2id[self.sep]
    
    @classmethod
    def build_vocab(cls, dataset, min_freq: int=5):
        counter = Counter(chain.from_iterable(
            sample['tokenized_context'] + sample['tokenized_question']
            for sample in tqdm(dataset, desc="Counting vocab")
        ))
    
        return cls([word for word, count in counter.items() if count >= min_freq])
    
    # id를 문자로
    def decode(self, token_ids: Sequence[int]):
        return [self.id2token[token_id] for token_id in token_ids]
    
    def sample2ids(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        context = [self.token2id.get(token, self.unk_id) for token in sample['tokenized_context']]
        question = [self.token2id.get(token, self.unk_id) for token in sample['tokenized_question']]
        
        # 스페셜 토큰과, 앞에 있는 질문을 제외한 context만 저장
        context = context[:self.max_length - len(question) - 3]
        input_ids = [self.cls_id] + question + [self.sep_id] + context + [self.sep_id]
        # +1은 앞의 cls 토큰, +2는 context 앞뒤의 sep 토큰
        token_type_ids = [0] * (len(question) + 1) + [1] * (len(context) + 2)
        
        # 스페셜 토큰들이 더해졌을 때, context안에 있는 answer의 position 구하기
        # 단순히 앞의 스페셜 토큰 cls와 question 그리고 sep 만큼만 더해주면 된다.
        if sample['answers'] is not None:
            # 답이 두 개 이상이더라도, 첫 번째 답만 보도록 한다.
            answer = sample['answers'][0]
            start = min(answer['start'] + len(question) + 2, self.max_length - 1)
            end = min(answer['end'] + len(question) + 2, self.max_length - 1)
        else:
            start = None
            end = None
            
        return {
            'guid': sample['guid'],
            'context': sample['original_context'],
            'question': sample['original_question'],
            'position': sample['context_position'],
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'start': start,
            'end': end
        }

data = TokenizedKoMRC.load("/Users/chanwoo/Downloads/trainForMrc.json")
train, val = TokenizedKoMRC.train_val_split(data)
indexer = Indexer.build_vocab(data)

#%%
class IndexerWrappedDataset:
    def __init__(self, dataset: TokenizedKoMRC, indexer: Indexer) -> None:
        self._dataset = dataset
        self._indexer = indexer
        
    def __len__(self) -> int:
        return len(self._dataset)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._indexer.sample2ids(self._dataset[index])
        sample['attention_mask'] = [1] * len(sample['input_ids'])
        
        return sample
#%%

indexed_train_dataset = IndexerWrappedDataset(train, indexer)
indexed_val_dataset = IndexerWrappedDataset(val, indexer)

sample = indexed_train_dataset[4]

print(sample['start'], sample['end'])
# print(sample['position'])
print(sample['context'])
start = sample['position'][sample['start'] - 2][0]
end = sample['position'][sample['end'] - 2][1]
# print(start, end)
print(sample['context'][start:end])
# %%
