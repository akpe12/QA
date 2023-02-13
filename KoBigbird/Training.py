wandb.init(project='KoBigbird', name='')
wandb.watch(model)

train_losses = []
dev_losses = []
step = 0
best_loss = 9999
epochs = 3

for epoch in range(epochs):
    print("Epoch", epoch)
    # Training
    model.train()
    running_loss = 0.
    losses = []
    progress_bar = tqdm(train_loader, desc='Train')

    for batch, start, end, answer_text in progress_bar:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)     
        start = start.to(device)
        end = end.to(device)
        output = model(input_ids = input_ids, attention_mask = attention_mask,start_positions = start, end_positions = end)

        loss = output.loss
        
        wandb.log({"Train_Loss": loss}) # train loss

        (loss / accumulation).backward()
        running_loss += loss.item()
        del batch, input_ids, attention_mask, start, end, loss

        step += 1
        if step % accumulation:
            continue

        clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        losses.append(running_loss / accumulation)
        running_loss = 0.
        progress_bar.set_description(f"Train - Loss: {losses[-1]:.3f}")
    train_losses.append(mean(losses))
    print(f"train score: {train_losses[-1]:.3f}")
        

    # Evaluation
    dev_loss = []
    model.eval()
    for batch, start, end, answer_text in val_loader:

        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        start = start.to(device)
        end = end.to(device)

        with torch.no_grad():
            outputs = model(input_ids = input_ids, attention_mask = attention_mask,start_positions = start, end_positions = end)
        loss = outputs.loss

        dev_loss.append(loss.item())
        del batch, input_ids, attention_mask, outputs, loss
    dev_losses.append(mean(dev_loss))
    print(f"Evaluation score: {dev_losses[-1]:.3f}")
    wandb.log({"Val_Loss": mean(dev_loss)})

    if best_loss >= dev_losses[-1]:
        best_loss = dev_losses[-1]
        model.save_pretrained('./')
        print('..Model____Save..')    
