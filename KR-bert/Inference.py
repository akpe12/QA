import csv

with torch.no_grad(), open('./KRBert_epoch3_1e-4#2.csv', 'w') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])

    rows = []
    for input, guid, context, position in tqdm(test_dataloader, "Testing"):
        input_ids, token_type_ids = input['input_ids'], input['token_type_ids']
        input.to(device)

        with torch.no_grad():
            outputs = model(**input)
        for i in range(len(guid)):            
        
            start_prob = outputs.start_logits[i][token_type_ids[i].bool()][1:-1].softmax(-1)
            end_prob = outputs.end_logits[i][token_type_ids[i].bool()][1:-1].softmax(-1)
            probability = torch.triu(start_prob[:, None] @ end_prob[None, :])
            index = torch.argmax(probability).item()
        
            start = index // len(end_prob)
            end = index % len(end_prob)

            start = position[i][start+1][0]
            end = position[i][end+1][1]

            inference = context[i][start : end]
            rows.append([guid[i], inference])

    writer.writerows(rows)
