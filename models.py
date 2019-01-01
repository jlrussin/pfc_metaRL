import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(LSTM, self).__init__()

        # Hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim + 1 #input: s_t,a_{t-1},r_{t-1}
        self.hidden_dim = hidden_dim

        # Parameters
        self.W_xi = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.W_hi = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_xf = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.W_hf = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_xc = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.W_hc = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.W_xo = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.W_ho = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.W_a = nn.Linear(self.hidden_dim, self.action_dim, bias=True)
        self.W_v = nn.Linear(self.hidden_dim, 1, bias=True)

        # Activations
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # Hidden states
        self.h = torch.zeros(self.hidden_dim)
        self.c = torch.zeros(self.hidden_dim)

    def forward(self,s,a_prev,r_prev):

        # Concatenate state, action and reward for input
        s = s.reshape(-1,self.state_dim)
        a_prev = a_prev.reshape(-1,self.action_dim)
        r_prev = r_prev.reshape(-1,1)
        x = torch.cat((s,a_prev,r_prev),dim=1)

        # Standard LSTM equations
        i = self.sigmoid(self.W_xi(x)+self.W_hi(self.h))
        f = self.sigmoid(self.W_xf(x)+self.W_hf(self.h))
        self.c = f*self.c + i*self.tanh(self.W_xc(x)+self.W_hc(self.h))
        o = self.sigmoid(self.W_xo(x) + self.W_ho(self.h))
        self.h = o*self.tanh(self.c)

        # Compute new distribution over actions and new value estimate
        probs = self.softmax(self.W_a(self.h))
        v = self.W_v(self.h)
        return probs,v

    def reinitialize(self):
        self.h = torch.zeros(self.hidden_dim)
        self.c = torch.zeros(self.hidden_dim)
