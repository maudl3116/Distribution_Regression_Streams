import torch
from torch.nn.utils.rnn import pack_padded_sequence as PACK
import torch.nn as nn
import torch.nn.functional as F

# class MIL_LSTM(nn.Module):
#     def __init__(self):
#         self.lstm = nn.LSTM(2, 1,batch_first=True)  # Note that "batch_first" is set to "True"
#
#     def forward(self, batch):
#         x, x_lengths, _ = batch
#         x_pack = PACK(x, x_lengths, batch_first=True)
#         output, hidden = self.lstm(x_pack)

class MIL_LSTM(nn.Module):
    #https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
    def __init__(self, input_dim, hidden_dim,layer_dim, output_dim):
        super(MIL_LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = out[:, -1, :]
        out = self.fc(out)
        # out.size() --> 100, 10
        return out

# class MIL_LSTM(nn.Module):
#     #https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
#     def __init__(self, input_size, output_size, hidden_dim, n_layers):
#         super(MIL_LSTM, self).__init__()
#
#         # Defining some parameters
#         self.hidden_dim = hidden_dim
#         self.n_layers = n_layers
#
#         # Defining the layers
#         # RNN Layer
#         self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
#         # Fully connected layer
#         self.fc = nn.Linear(hidden_dim, output_size)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#
#         # Initializing hidden state for first input using method defined below
#         hidden = self.init_hidden(batch_size)
#
#         # Passing in the input and hidden state into the model and obtaining outputs
#         out, hidden = self.rnn(x, hidden)
#
#         # Reshaping the outputs such that it can be fit into the fully connected layer
#         out = out.contiguous().view(-1, self.hidden_dim)
#         out = self.fc(out)
#
#         return out
#
#     def init_hidden(self, batch_size):
#         # This method generates the first hidden state of zeros which we'll use in the forward pass
#         # We'll send the tensor holding the hidden state to the device we specified earlier as well
#         hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
#         return hidden

class MIL_RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=1):
        super(MIL_RNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define the LSTM layer
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, self.num_layers,batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.linear2 = nn.Linear(self.hidden_dim, output_dim)
    # def init_hidden(self):
    #     # This is what we'll initialise our hidden state as
    #     return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
    #             torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, batch):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).

        # Initializing hidden state for first input using method defined below
        batch_size = batch.size(0)

        hidden = self.init_hidden(batch_size)
        #print(x_pack.shape)
        rnn_out, self.hidden = self.rnn(batch,hidden)#self.lstm(batch.view(len(batch), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction

        y_pred = self.linear(rnn_out[:, -1, :])#unpacked.transpose(0,1)[-1]
        return y_pred

    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        return hidden