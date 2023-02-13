class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.question, self.context, self.guid, self.positions = self.make_dataset(dataset)

    def make_dataset(self, dataset):
        context, question, guid, positions = [], [], [], []
        for i, data in enumerate(dataset) :          
          context.append(data['context'])
          question.append(data['question'])
          guid.append(data['guid'])
          positions.append(remove_blank_tokens(tokenizer(data["context"], return_offsets_mapping=True).offset_mapping))
        return question, context, guid, positions
    
    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.guid[idx], self.context[idx], self.question[idx], self.positions[idx]

test_dataset = KoMRC.load('/.json')
test_dataset = CustomDataset(test_dataset)        
