import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch_geometric.nn import SGConv
cudnn.deterministic = True  
cudnn.benchmark = True

class Regularizer(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(Regularizer, self).__init__()
        self.den1 = torch.nn.Linear(input_dim, hidden_dim1)
        self.den1.bias.data.fill_(0.0)
        self.den1.weight.data = torch.normal(0.0, 0.001, [hidden_dim1, input_dim])
        self.den2 = torch.nn.Linear(hidden_dim1, hidden_dim2)
        self.den2.bias.data.fill_(0.0)
        self.den2.weight.data = torch.normal(0.0, 0.001, [hidden_dim2, hidden_dim1])
        self.output = torch.nn.Linear(hidden_dim2, 1)
        self.output.bias.data.fill_(0.0)
        self.output.weight.data = torch.normal(0.0, 0.001, [1,hidden_dim2])
        self.act = torch.sigmoid
    def forward(self, inputs):
        dc_den1 = self.act(self.den1(inputs))
        dc_den2 = torch.sigmoid((self.den2(dc_den1)))
        output = self.output(dc_den2)
        return output


class VariationalEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(VariationalEncoder,self).__init__()
        self.conv1 = SGConv(in_channels, hidden_channels)
        self.conv_mu = SGConv(hidden_channels, out_channels)
        self.conv_logstd = SGConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)   