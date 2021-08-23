# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 06:52:38 2021

@author: karlo
"""

import torch
import torch.nn as nn




class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_size, kernel_size, bias = True,):
        """
        Initialize ConvLSTM Cell.

        Parameters
        ----------
        input_dim : int
            Number of color channels in input raster. (1 for MODIS)
        hidden_size : int
            Number of neurons in the hidden layer.
        kernel_size : (int, int)
            Convolutional kernel size.
        bias : bool
            Adding bias.

        Returns
        -------
        new hidden_state, new cell_state
        """
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        
        self.kernel_size = kernel_size
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.bias = bias
                
        self.conv2d = nn.Conv2d(in_channels = self.input_dim + self.hidden_size, 
                                out_channels = 4 * self.hidden_size,
                                kernel_size = self.kernel_size,
                                padding = self.padding,
                                bias = self.bias)
        
    def forward(self, input_tensor, current_state):
        
        h_state, c_state = current_state # hidden and cell states
        
        combined = torch.cat((input_tensor, h_state), dim=1) # concatenate xt and h(t-1)
        
        conv_comb = self.conv2d(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(conv_comb, self.hidden_size, dim=1)
        # splitting the tensor to 4 gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # calculating gate outputs
        c_next = f * c_state + i * g # calculating new cell_state
        h_next = o * torch.tanh(c_next) # calculating new hidden state
        
        return h_next, c_next

    def init_hidden(self, batch_size, input_shape):        
        return (torch.zeros(batch_size, self.hidden_size, input_shape[0], 
                            input_shape[1], device=self.conv2d.weight.device),
                torch.zeros(batch_size, self.hidden_size, input_shape[0], 
                            input_shape[1], device=self.conv2d.weight.device))
        

class ConvLSTM(nn.Module):
    
    """
    Parameters:
        input_dim: Number of color channels in input raster.
        hidden_size: Number of neurons in the hidden layer.
        kernel_size: Convolutional kernel size.
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_size, kernel_size, num_layers, 
                 batch_first = True, bias = True, return_all_layers = False):
        super(ConvLSTM, self).__init__()
        
        self._check_kernel_size_consistency(kernel_size)
        
        # Checking if both kernel_size and hidden_size are lists having lentgh == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_size = self._extend_for_multilayer(hidden_size, num_layers)
        
        if not len(kernel_size) == len(hidden_size) == num_layers:
            raise ValueError("Inconsistent list length.")
            

        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        cell_list = list()
        for l in range(0, self.num_layers):
            curr_input_dim = self.input_dim if l == 0 else self.hidden_size[l - 1]
            
            cell_list.append(ConvLSTMCell(input_dim = curr_input_dim,
                                          hidden_size = self.hidden_size[l],
                                          kernel_size = self.kernel_size[l],
                                          bias = self.bias))
            
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None, prints = True):
        """
        Parameters
        ----------
        input_tensor: 
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: 
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """ 
        
        
        if not self.batch_first:
            # (timesteps, batch_size, channels, height, weight) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        b, seq_len, _, h, w = input_tensor.size()
        if prints: print("input_tensor shape:", input_tensor.shape, "[num_batches, seq_len, channels, height, width]")
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward, send image_size here
            hidden_state = self.get_init_hidden(batch_size = b, 
                                                input_size = (h, w))
        
        layer_output_list = list()
        last_state_list = list()
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = list()
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 current_state = [h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            
            if prints: print("layer_output shape:", layer_output.shape, "[num_batches, seq_len, channels, height, width]")
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list [-1:]
        
        if prints: print("layer_output shape:", last_state_list[0][0].shape, "")   
        return layer_output_list, last_state_list

    
    def get_init_hidden(self, batch_size, input_size,):
        init_states = list()
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, input_size,))
        return init_states
        

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or 
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError(" 'kernel_size' must be a tuple or list of tuples.")
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param 