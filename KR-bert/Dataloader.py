def make_dataloader(dataset, batch_size, type_name) :
  dataloader = DataLoader(
      dataset,
      batch_size = batch_size,
      collate_fn = custom_collate_fn,
      num_workers = 2,
      shuffle= True if type_name == "train" else False
  )
  print(f'batch_size : {batch_size}')
  return dataloader

train_dataset = CustomDataset(new_train_dataset)
valid_dataset = CustomDataset(new_val_dataset)

batch_size = 64
accumulation = 4

train_dataloader = make_dataloader(train_dataset, batch_size=batch_size // accumulation, type_name="train")
valid_dataloader = make_dataloader(valid_dataset, batch_size=batch_size // accumulation, type_name="valid")
