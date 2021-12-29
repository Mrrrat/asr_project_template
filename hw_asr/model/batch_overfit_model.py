from torch import nn
from torch.nn import Sequential
from hw_asr.base import BaseModel

class BatchOverfitModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, num_layers=2,*args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        #self.gru = nn.GRU(n_feats, fc_hidden, num_layers=num_layers,  bidirectional=True)
        self.lstm = nn.LSTM(n_feats, hidden_size=fc_hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(fc_hidden * 2, n_class)

    def forward(self, spectrogram, *args, **kwargs):
        x, _ = self.lstm(spectrogram.transpose(1, 2))
        x = self.fc(x)
        return  {"logits": x}

    def transform_input_lengths(self, input_lengths):
        return input_lengths