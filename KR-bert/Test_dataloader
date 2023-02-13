def custom_collate_fn(batch):
    global tokenizer
    guid_list, context_list, question_list, positions = [], [], [], []
    for _guid, _context, _question, _position in batch:
        question_list.append(_question)
        context_list.append(_context)
        guid_list.append(_guid)
        positions.append(_position)
        
    tensorized_input = tokenizer(    # 정답 잘림 방지를 위해 max_len과 truncation 제외?
        question_list, context_list,
        add_special_tokens=True,
        padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
        max_length=512,
        truncation=True,
        return_tensors='pt'
    )

    # context_tokens = tokenizer(
    #     context_list,
    #     add_special_tokens=False,
    #     return_tensors=None
    # ).input_ids

    return tensorized_input, guid_list, context_list, positions
    
def make_dataloader(dataset, batch_size, type_name) :
  dataloader = DataLoader(
      dataset,
      batch_size = batch_size,
      collate_fn = custom_collate_fn,
      num_workers = 2,
      shuffle=True if type_name == "train" else False
  )
  print(f'batch_size : {batch_size}')
  return dataloader    
