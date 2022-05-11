import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

class AggregateModule(nn.Module):
    def __init__(self, hidden_size, restrict_update=False, normalize_map_features=False):
        super().__init__()
        self.restrict_update = restrict_update
        self.normalize_map_features = normalize_map_features

    def forward(self, x, hidden):
        """
        Inputs:
            x      - (bs, hidden_size)
            hidden - (bs, hidden_size)
        """
        x1 = self._compute_forward(x, hidden) # (bs, hidden_size)
        # Retain hidden state wherever no updates are needed
        if self.restrict_update:
            print('restrict_update occurs')
            x_mask = (x != 0).float()
            x1 =  x_mask * x1 + (1-x_mask) * hidden
        # Normalize map features if necessary
        if self.normalize_map_features:
            print('normalize_map_features occurs')
            x1 = F.normalize(x1 + 1e-10, dim=1)
        return x1

    def _compute_forward(self, x, hidden):
        raise NotImplementedError

class GRUAggregate(AggregateModule):
    def __init__(
        self,
        hidden_size,
        restrict_update=False,
        normalize_map_features=False
    ):
        super().__init__(
            hidden_size,
            restrict_update=restrict_update,
            normalize_map_features=normalize_map_features
        )
        self.main = nn.GRU(hidden_size, hidden_size, num_layers=1) #default: batch_first is False
        # All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        # where :math:`k = \frac{1}{\text{hidden\_size}}`
        for name, param in self.main.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def _compute_forward(self, x, hidden):
        x1, _ = self.main(x.unsqueeze(0).contiguous(), hidden.unsqueeze(0).contiguous())
        return x1[0]

class GRUAggregate_HiddenNorm(AggregateModule):
    def __init__(
        self,
        hidden_size,
        restrict_update=False,
        normalize_map_features=False,
        norm_h=False
    ):
        super().__init__(
            hidden_size,
            restrict_update=restrict_update,
            normalize_map_features=normalize_map_features
        )
        self.main = GRUModel(hidden_size, hidden_size, layer_dim=1, bias=True,norm_h=norm_h) #default: batch_first is False
        # All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        # where :math:`k = \frac{1}{\text{hidden\_size}}`
        for name, param in self.main.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def _compute_forward(self, x, hidden):
        x1, _ = self.main(x.unsqueeze(0).contiguous(), hidden.unsqueeze(0).contiguous())
        return x1[0]

class ConvGRUAggregate(AggregateModule):
    def __init__(
        self,
        hidden_size,
        restrict_update=False,
        normalize_map_features=False,
        norm_h = False
    ):
        super().__init__(
            hidden_size,
            restrict_update=restrict_update,
            normalize_map_features=normalize_map_features
        )
        self.main = ConvGRUModel(hidden_size, hidden_size, layer_dim=1, bias=True, norm_h=norm_h) #default: batch_first is False
        # All the weights and biases are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`
        # where :math:`k = \frac{1}{\text{hidden\_size}}`
        for name, param in self.main.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def _compute_forward(self, x, hidden):
        x1, _ = self.main(x.unsqueeze(0).contiguous(), hidden.unsqueeze(0).contiguous())
        return x1[0]

class GRUCell(nn.Module):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True, norm_h=False):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.norm_h = norm_h
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        gate_x = self.x2h(x) 
        gate_h = self.h2h(hidden)
        
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        
        
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        # hy = hidden + inputgate * (newgate - hidden)
        if self.norm_h:
            hy = 2* torch.softmax(hy,dim=-1)-1
        return hy

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim=1, bias=True, norm_h=False):
        super(GRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bias = bias
        self.norm_h = norm_h

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(GRUCell(self.input_dim, self.hidden_dim,self.bias,self.norm_h))

        for l in range(1,self.layer_dim):
            self.rnn_cell_list.append(GRUCell(self.hidden_dim, self.hidden_dim,self.bias,self.norm_h))
         
    
    def forward(self, x,hx=None):
        
        # Initialize hidden state with zeros
        #######################
        #  USE GPU FOR MODEL  #
        #######################
        # Inputs:
        #       x: of shape (L,batch_size, input_size)
        #       hx: of shape (L,batch_size, hidden_size)
        # Output:
        #       hy: of shape (L,batch_size, hidden_size)
        if hx is None:
            if torch.cuda.is_available():
                h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim).cuda()
            else:
                h0 = torch.zeros(self.layer_dim, x.size(1), self.hidden_dim)
        else:
            h0 =hx      

        outs = []        
        hidden = []
        for layer in range(self.layer_dim):
            hidden.append(h0[layer, :, :]) # [[batch_size,hidden_size],[],...]

        for t in range(x.size(0)): #Sequence Length: L

            for layer in range(self.layer_dim):

                if layer == 0:
                    #GRU(x=[batch_size,input_size],h=[batch_size,hidden_size])
                    hidden_l = self.rnn_cell_list[layer](x[t, :, :], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l #[Layer,batch_size,hidden_size]

            outs.append(hidden_l) #[L,batch_size,hidden_size]

        # Take only last time step. Modify for seq to seq
        # outs = outs[[-1]]

        return outs, hidden
 
class ConvGRUCell(nn.Module):
    def __init__(self,input_dim,hidden_dim, kernel_size=3, bias=True, norm_h=False):
        super(ConvGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.norm_h = norm_h

        self.convx2h = nn.Conv2d(in_channels=self.input_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.convh2h = nn.Conv2d(in_channels=self.hidden_dim,
                              out_channels=3 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self,x, hidden):
        '''
        input:
            x: b,cls,s1,s2
            hidden: b,h,s1,s2
        output: 
            hy: b,h,s1,s2
        '''
        gate_x = self.convx2h(x)  #(b,3*h,s1,s2)
        gate_h = self.convh2h(hidden)#(b,3*h,s1,s2)

        i_r, i_i, i_n = gate_x.chunk(3, 1) #(b,h,s1,s2)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))

        hy = newgate + inputgate * (hidden - newgate)
        # hy = hidden + inputgate * (newgate - hidden)
        if self.norm_h:
            hy = 2* torch.softmax(hy,dim=1)-1

        return hy

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=self.convx2h.weight.device)

class ConvGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, layer_dim=1, bias=True, norm_h=False):
        super(ConvGRUModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.layer_dim = layer_dim
        self.bias = bias
        self.norm_h = norm_h

        self.rnn_cell_list = nn.ModuleList()

        self.rnn_cell_list.append(ConvGRUCell(self.input_dim, self.hidden_dim,self.kernel_size,self.bias,self.norm_h))

        for l in range(1,self.layer_dim):
            self.rnn_cell_list.append(ConvGRUCell(self.hidden_dim, self.hidden_dim,self.kernel_size,self.bias,self.norm_h))
         
    
    def forward(self, x,hx=None):
        
        # Initialize hidden state with zeros
        # Inputs:
        #       x: of shape  (T, b, cls, h,w)
        #       hx: of shape (L, b, hidden, h,w)
        # Output:
        #       hy: of shape (L, b, hidden, h, w)
        _, b, _, h, w = x.size()
        if hx is None:
            h0 = self._init_hidden(batch_size=b, image_size=(h, w))
        else:
            h0 =hx      

        outs = []        
        hidden = []
        for layer in range(self.layer_dim):
            hidden.append(h0[layer]) # [[b,hidden,h,w],[],...]

        for t in range(x.size(0)): #Sequence Length: L

            for layer in range(self.layer_dim):

                if layer == 0:
                    #GRU(x=[batch_size,input_size],h=[batch_size,hidden_size])
                    hidden_l = self.rnn_cell_list[layer](x[t], hidden[layer])
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1],hidden[layer])
                hidden[layer] = hidden_l #[Layer,batch_size,hidden_size,h,w]

            outs.append(hidden_l) #[T,batch_size,hidden_size,h,w]

        # Take only last time step. Modify for seq to seq
        # outs = outs[[-1]]

        return outs, hidden

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.rnn_cell_list[i].init_hidden(batch_size, image_size))
        return init_states

