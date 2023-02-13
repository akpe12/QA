import json
from typing import Dict, Any
import random

class KoMRC:
    def __init__(self, data, indices) -> None:
        self._data = data
        self._indices = indices

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        indices = []
        for data_id, document in enumerate(data['data']):
            for para_id, paragraph in enumerate(document['paragraphs']):
                for question_id, _ in enumerate(paragraph['qas']):
                    indices.append((data_id, para_id, question_id))
        
        return cls(data, indices) # KoMRC의 생성자 호출!
    
    @classmethod
    def train_val_split(cls, dataset, val_ratio: float=.1, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * val_ratio) :]
        val_indices = indices[: int(len(indices) * val_ratio)]
        
        return cls(dataset._data, train_indices), cls(dataset._data, val_indices)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        # self._indices 안에는 data_id, para_id, question_id가 들어있다.
        # return 시켜야 하는 정보는, context/ question/ answer/ guid
        
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]
        
        # paragraph 안에는, context/ qas
        context = paragraph['context']
        qa = paragraph['qas'][q_id]
        
        question = qa['question']
        guid = qa['guid']
        answers = qa['answers']
        
        return {
            'guid': guid,
            'context': context,
            'question': question,
            'answers': answers
        }
        
    def __len__(self) -> int:
        return len(self._indices)

data = KoMRC.load("/Users/chanwoo/Downloads/trainForMrc.json")
# train, val = KoMRC.train_val_split(data)