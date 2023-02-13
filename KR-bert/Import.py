!pip install transformers
!pip install fastcore
from fastcore.all import patch_to
import unicodedata
from transformers import BertModel, BertTokenizer, BertTokenizerFast, BertConfig, BertPreTrainedModel, BertForQuestionAnswering

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm, tqdm_notebook

import matplotlib.pyplot as plt
# %pwd
%cd "/content/drive/MyDrive/sam"
from sam import SAM
%cd "/content"
from typing import Dict, Tuple, Any, List
import json
import random
import csv

model_name = "snunlp/KR-BERT-char16424"

!pip install wandb==0.9.7
import wandb
wandb.login()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device, torch.cuda.device_count()

tokenizer = BertTokenizerFast.from_pretrained(model_name)
