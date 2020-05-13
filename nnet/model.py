import torch
from torch.nn.utils.rnn import pack_padded_sequence as PACK
import torch.nn as nn


# class MIL_LSTM(nn.Module):
#     def __init__(self):
#         self.lstm = nn.LSTM(2, 1,batch_first=True)  # Note that "batch_first" is set to "True"
#
#     def forward(self, batch):
#         x, x_lengths, _ = batch
#         x_pack = PACK(x, x_lengths, batch_first=True)
#         output, hidden = self.lstm(x_pack)


class MIL_LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=2):
        super(MIL_LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    # def init_hidden(self):
    #     # This is what we'll initialise our hidden state as
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, batch):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        x, x_lengths = batch
        x_pack = PACK(x, x_lengths, batch_first=True)

        lstm_out, self.hidden = self.lstm(x_pack)#self.lstm(batch.view(len(batch), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        y_pred = self.linear(unpacked.transpose(0,1)[-1])#.view(self.batch_size, -1))

        return y_pred


class MIL_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=1):
        super(MIL_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    # def init_hidden(self):
    #     # This is what we'll initialise our hidden state as
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, batch):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        x, x_lengths = batch
        x_pack = PACK(x, x_lengths, batch_first=True)

        rnn_out, self.hidden = self.rnn(x_pack)#self.lstm(batch.view(len(batch), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)

        y_pred = self.linear(unpacked.transpose(0,1)[-1])#.view(self.batch_size, -1))

        return y_pred