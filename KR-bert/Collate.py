def custom_collate_fn(batch):
    global tokenizer
    question_list, context_list, answer_start, answer_text, after_list, positions = [], [], [], [], [], []
    for _question, _context, _start, _text, _position in batch:
        question_list.append(_question)
        context_list.append(_context)
        after_list.append(_context[:_start])
        answer_start.append(_start)
        answer_text.append(_text)
        positions.append(_position)

    tensorized_input = tokenizer(    # 정답 잘림 방지를 위해 max_len과 truncation 제외?
        question_list, context_list,
        add_special_tokens=True,
        padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
        max_length=512,
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
        start.append(len(label) + 2) # 2 => CLS, SEP(중간)
        end.append(len(label) + len(answer) + 1) # 1 => SEP(마지막)
    start = torch.tensor(start).long()
    end = torch.tensor(end).long()
    return tensorized_input, start, end, answer_text, positions, context_list
