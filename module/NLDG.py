from torch import nn
import torch


class NLDG(nn.Module):
    def __init__(self, fcA_dim, hidden_size, num_layers):
        super(NLDG, self).__init__()
        self.fcA_dim = fcA_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.fcA = nn.LazyLinear(fcA_dim)
        self.sigmoid = nn.Sigmoid()

        self.gru = nn.GRU(
            input_size=1 + fcA_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

        self.fcB = nn.Linear(hidden_size, 1)

        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param)
            else:
                nn.init.zeros_(param)

    def init_hidden(self, batch_size, device="cuda"):
        return (torch.rand(self.num_layers, batch_size, self.hidden_size) * 0.01).to(device)

    def forward(self, x, x_news, h0):
        # x -> [batch_size, seq_len]
        x = x.unsqueeze(-1)
        # x -> [batch_size, seq_len, 1]
        # x_news -> [batch_size, seq_len, x_news_dim]
        nf = self.fcA(x_news)
        # nf -> [batch_size, seq_len, fc1_dim]
        nf = self.sigmoid(nf)
        x = torch.cat([x, nf], dim=-1)
        # x -> [batch_size, seq_len, 1 + fc1_dim]
        o, h1 = self.gru(x, h0)
        # o -> [batch_size, seq_len, hidden_size]
        o = o[:, -1, :]
        # o -> [batch_size, hidden_size]
        o = self.fcB(o)
        # o -> [batch_size, 1]
        o.squeeze_(dim=-1)
        # o -> [batch_size]
        return o, h1