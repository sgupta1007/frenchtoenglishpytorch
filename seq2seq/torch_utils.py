import torch

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')
MAX_LENGTH=10
SOS_TOKEN=0
EOS_TOKEN=1
teacher_forcing_ratio=0.5
hidden_size=256