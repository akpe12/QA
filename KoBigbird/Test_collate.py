def test_collate_fn(batch):
    global tokenizer
    question_list, context_list, guid_list= [], [], []

    for _question, _context, _guid in batch:
        question_list.append(_question)
        context_list.append(_context)
        guid_list.append(_guid)

    tensorized_input = tokenizer(    
        question_list, context_list,
        add_special_tokens=True,
        padding="max_length",  
        max_length=2048,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors='pt'
        
    )

    return tensorized_input, guid_list, context_list
