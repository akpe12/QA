def make_dataloader(dataset, batch_size, mode='train') :
  dataloader = DataLoader(
      dataset,
      batch_size = batch_size,
      shuffle = True if mode == 'train' else False,
      collate_fn = custom_collate_fn
  )
  print(f'batch_size : {batch_size}')
  return dataloader
