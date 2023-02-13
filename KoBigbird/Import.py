!pip install transformers
!pip install datasets

from datasets import load_dataset, load_metric#, list_metrics

from transformers import AutoTokenizer, BertConfig, BertPreTrainedModel, AutoModelForQuestionAnswering
from statistics import mean
import pandas as pd
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import QuestionAnsweringModelOutput
import os
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm, tqdm_notebook
import warnings
import time
import random
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings(action='ignore')

tokenizer = AutoTokenizer.from_pretrained(
    "monologg/kobigbird-bert-base",
    use_fast=True # Whether or not to try to load the fast version of the tokenizer.
    )

!pip install wandb==0.9.7
import wandb
wandb.login()

# train_df = pd.read_csv('/content/drive/MyDrive/MRC_with_AI-hub/train_dataset.csv',encoding = 'utf8')
# val_df = pd.read_csv('/content/drive/MyDrive/MRC_with_AI-hub/val_dataset.csv',encoding = 'utf8')
