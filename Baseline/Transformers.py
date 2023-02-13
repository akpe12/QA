import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

class BertForQuestionAnswering(BertPreTrainedModel):
    # bert에다가 liear projection(H x 1) 2개 달기
    # BertModel(config, add_pooling_layer=bool)
    # -> add_pooling_layer를 True로 하면 CLS token만 들고 오고, False로 하면 전체 토큰 들고 온다.
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        self.start_linear = nn.Linear(config.hidden_size, 1)
        self.end_linear = nn.Linear(config.hidden_size, 1)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        
        start_logits = self.start_linear(outputs.last_hidden_state).squeeze(-1)
        end_logits = self.end_linear(outputs.last_hidden_state).squeeze(-1)
        
        return start_logits, end_logits