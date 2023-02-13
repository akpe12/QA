from typing import List, Tuple, Dict, Any
import json
import random

class TestKoMRC:
    def __init__(self, data, indices: List[Tuple[int, int, int]]):
        self._data = data
        self._indices = indices

    # Json을 불러오는 메소드
    @classmethod
    def load(cls, file_path: str):
        with open(file_path, 'r', encoding='utf-8') as fd:
            data = json.load(fd)

        indices = []
        for d_id, document in enumerate(data['data']):
            for p_id, paragraph in enumerate(document['paragraphs']):
                for q_id, _ in enumerate(paragraph['qas']):
                    indices.append((d_id, p_id, q_id))
        
        return cls(data, indices)

    # 데이터 셋을 잘라내는 메소드
    @classmethod
    def split(cls, dataset, eval_ratio: float=.1, seed=42):
        indices = list(dataset._indices)
        random.seed(seed)
        random.shuffle(indices)
        train_indices = indices[int(len(indices) * eval_ratio):]
        eval_indices = indices[:int(len(indices) * eval_ratio)]

        return cls(dataset._data, train_indices), cls(dataset._data, eval_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        d_id, p_id, q_id = self._indices[index]
        paragraph = self._data['data'][d_id]['paragraphs'][p_id]

        
        qa = paragraph['qas'][q_id]

        guid = qa['guid']
        question = qa['question']
        context = paragraph['context']
        return {
            'guid': guid,
            'context': context,
            'question': question
        }

    def __len__(self) -> int:
        return len(self._indices)

class TestDataset(Dataset):
    def __init__(self, dataset):
        self.question, self.context, self.guid= self.make_dataset(dataset)

    def make_dataset(self, dataset):
        context, question, guid = [], [], []
        for i, data in enumerate(dataset) :
          context.append(data['context'])
          question.append(data['question'])
          guid.append(data['guid'])
        return question, context, guid
        
    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        return self.question[idx], self.context[idx] , self.guid[idx]
    
