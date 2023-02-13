def custom_collate_fn(batch):
    global tokenizer
    question_list, context_list, answer_start, answer_text, after_list = [], [], [], [], []

    for _question, _context, _start, _text in batch:
        question_list.append(_question)
        context_list.append(_context)
        after_list.append(_context[:_start])
        answer_start.append(_start)
        answer_text.append(_text)

    tensorized_input = tokenizer(    
        question_list, context_list,
        add_special_tokens=True,
        padding="max_length",  
        max_length=2048,
        truncation=True,
        return_tensors='pt'
    )

    # answer_start token의 위치를 찾기 위해 list로 반환
    label_tokens = tokenizer(
        question_list, after_list,
        add_special_tokens=False,
        return_tensors=None
    ).input_ids

    # answer text token만 변환
    answer_tokens = tokenizer(
        answer_text,
        add_special_tokens=False,
        return_tensors=None
    ).input_ids

    start = []
    end = []
    for label, answer in zip(label_tokens,answer_tokens):
        
        start.append(len(label) + 2)
        end.append(len(label) + len(answer) + 1)

    start = torch.LongTensor(start)
    end = torch.LongTensor(end)
    return tensorized_input, start, end, answer_text
