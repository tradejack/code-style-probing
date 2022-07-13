from transformers import PLBartModel, PLBartTokenizer
from torch import nn
from tqdm import tqdm
from copy import deepcopy

from transformers import PLBartTokenizer, PLBartModel
import torch

tokenizer = PLBartTokenizer.from_pretrained("uclanlp/plbart-base")
model = PLBartModel.from_pretrained("uclanlp/plbart-base")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state

# the plan here is to initialize a pretrained instance of PLBart and then steal the weights from the model's 
# encoder and decoder blocks. I've had issues loading in the model weights but I think thats bc my remote
# home directory is still messed up. 

class Test(nn.Module):
    def __init__(self, model):
        # Takes a pretrained, weights-loaded instance of a PLBartModel object
        # Other PLBart variations won't work bc they don't have encoder/decoder attributes
        super().__init__()
        self.enc = deepcopy(model.encoder)
        self.dec = deepcopy(model.decoder)
        self.disc = nn.Linear(1)
    
    def forward(self):
        # pass encoder -> decoder -> disc
        pass



