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
            'answers': answers,
        }
        
    def __len__(self) -> int:
        return len(self._indices)

# for pandas
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.question, self.context, self.answer_start, self.answer_text = self.make_dataset(dataset)

    def make_dataset(self, dataset):
        context, question, answer_start, answer_text = list(dataset.context), list(dataset.question), list(dataset.answer_start), list(dataset.answer_text)
        return question, context, answer_start, answer_text
        
    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        return self.question[idx], self.context[idx], self.answer_start[idx], self.answer_text[idx]
train_dataset = CustomDataset(train_df)
val_dataset = CustomDataset(val_df)
      
# for list
class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.question, self.context, self.answer_start, self.answer_text = self.make_dataset(dataset)

    def make_dataset(self, dataset):
        context, question, answer_start, answer_text = [], [], [], []
        for i, data in enumerate(dataset) :
          start = data['answers'][0]['answer_start']
        
          if start == None : # 답이 없을 때
            print(i, data)
            continue        
          answer_start.append(start)
          answer_text.append(data['answers'][0]['text'])
          context.append(data['context'])
          question.append(data['question'])
        return question, context, answer_start, answer_text

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.question[idx], self.context[idx], self.answer_start[idx], self.answer_text[idx]
train_dataset = CustomDataset(train_dataset)
val_dataset = CustomDataset(val_dataset)
