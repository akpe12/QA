# start, end 지점 필요!

import os
from statistics import mean

import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

wandb.init(project='KRbert-QA', name='')

# os.makedirs('dump', exist_ok=True)
train_step_losses = []
dev_step_losses = []
train_losses = []
dev_losses = []
lowest_dev_loss = 9999

epochs = 3
step = 0

for epoch in range(epochs):
    print("Epoch", epoch)
    # Training
    running_loss = 0.
    losses = []
    progress_bar = tqdm(train_dataloader, desc='Train')
    for input, start, end, answer_text, positions, context in progress_bar:
        model.train()
        del answer_text, positions, context
        input_ids = input.input_ids.to(device)
        attention_mask = input.attention_mask.to(device)
        start = start.cuda()
        end = end.cuda()
        
        outputs = model(input_ids = input_ids, attention_mask = attention_mask, start_positions=start, end_positions=end)
        loss = outputs.loss
        (loss / accumulation).backward()
        running_loss += loss.item()
        del input, start, end, loss, input_ids, attention_mask

        # start_logits, end_logits = model(**input)
        # loss = F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)        
        # (loss / accumulation).backward()
        # running_loss += loss.item()
        # del input, start, end, start_logits, end_logits, loss
        
        step += 1
        if step % accumulation: # step % acc == 0이 아니면 다시 backward하러 돌아가게끔
            continue

        clip_grad_norm_(model.parameters(), max_norm=1.)
        optimizer.step()
        scheduler.step(epoch+1)
        optimizer.zero_grad(set_to_none=True)

        train_step_losses.append(running_loss / accumulation)
        losses.append(running_loss / accumulation)
        running_loss = 0.
        progress_bar.set_description(f"Train - Loss: {losses[-1]:.3f}")
        wandb.log({"Train Loss": losses[-1]})
    train_losses.append(mean(losses))
    print(f"train score: {train_losses[-1]:.3f}")

        # if step % 1024:
        # Evaluation
    val_losses = []
    for input, start, end, answer_text, positions, context in tqdm(valid_dataloader, desc="Evaluation"):
        model.eval()
        del answer_text, positions, context
        input_ids = input.input_ids.to(device)
        attention_mask = input.attention_mask.to(device)
        start = start.cuda()
        end = end.cuda()
        
        with torch.no_grad():
            outputs = model(input_ids = input_ids, attention_mask = attention_mask, start_positions=start, end_positions=end)
        loss = outputs.loss

        # with torch.no_grad():
        #     start_logits, end_logits = model(**input)
        # loss = F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)

        dev_step_losses.append(loss.item())
        val_losses.append(loss.item())
        wandb.log({"Valid_loss": mean(dev_step_losses)}) # valid loss
        del input, start, end, loss, input_ids, attention_mask
    dev_losses.append(mean(dev_step_losses))
    print(f"Evaluation score: {dev_losses[-1]:.3f}")

    if lowest_dev_loss > dev_losses[-1]:                    
        lowest_dev_loss = dev_losses[-1]
        # torch.save(model.state_dict(), "./")
        model.save_pretrained(f'./')
        # model.train()

    # wandb.log({"Acc": acc}) # acc
