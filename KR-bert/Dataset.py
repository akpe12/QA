def remove_blank_tokens(offsets):
    offsets.pop(0)
    offsets.pop(-1)

    return offsets
    
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
            'positions': []
        }
        
    def __len__(self) -> int:
        return len(self._indices)

dataset = KoMRC.load("/content/drive/MyDrive/trainForMrc.json")
train_dataset, val_dataset = KoMRC.train_val_split(dataset)


train_dataset_ = list(train_dataset)
val_dataset_ = list(val_dataset)
def set_context(dataset):

    for i in range(len(dataset)):
        guid = dataset[i]['guid']
        context = dataset[i]['context']
        question = dataset[i]['question']
        answers = dataset[i]['answers']

        after = context[: answers[0]['answer_start']]
        answer_text = answers[0]['text']

        label_tokens = tokenizer(
            question, after,
            add_special_tokens=False,
            return_tensors=None
        ).input_ids

        # answer text token만 변환
        answer_tokens = tokenizer(
            answer_text,
            add_special_tokens=False,
            return_tensors=None
        ).input_ids

        question_tokens = tokenizer(
            question,
            add_special_tokens=False,
            return_tensors=None
        ).input_ids

        context_tokens = tokenizer(
            context,
            add_special_tokens=False,
            return_tensors=None
        ).input_ids
        
        start = len(label_tokens) + 2
        end = len(label_tokens) + len(answer_tokens) + 1
        # max_length - question 토큰 개수 - token개수 3 = 가능한 context의 길이

        max_length = 512
        max_context = max_length - len(question_tokens) - 3 # 가능 context 토큰 길이

        # context 밖에 정답이 위치한 경우
        # context의 맨 마지막이 정답의 end가 되도록
        if max_length - 3 < end:
            start -= (len(question_tokens) + 2)
            end -= (len(question_tokens) + 1)

            context_position = remove_blank_tokens(tokenizer(context, return_offsets_mapping=True).offset_mapping)
            answer_context = context[context_position[start - 350][0]: ]
            answer_start = answer_context.find(context[context_position[start][0] : context_position[end][1]])

            dataset[i]['context'] = answer_context
            dataset[i]['answers'][0]['answer_start'] = answer_start
            context = answer_context
        dataset[i]['positions'] = remove_blank_tokens(tokenizer(context, return_offsets_mapping=True).offset_mapping)
    
    return dataset
new_train_dataset = set_context(train_dataset_)
new_val_dataset = set_context(val_dataset_)

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.question, self.context, self.answer_start, self.answer_text, self.positions = self.make_dataset(dataset)

    def make_dataset(self, dataset):
        context, question, answer_start, answer_text, positions = [], [], [], [], []
        for i, data in enumerate(dataset) :
          start = data['answers'][0]['answer_start']
        
          if start == None : # 답이 없을 때
            print(i, data)
            continue        
          answer_start.append(start)
          answer_text.append(data['answers'][0]['text'])
          context.append(data['context'])
          question.append(data['question'])
          positions.append(data['positions'])
        return question, context, answer_start, answer_text, positions

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.question[idx], self.context[idx], self.answer_start[idx], self.answer_text[idx], self.positions[idx]
