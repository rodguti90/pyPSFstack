from turtle import forward
import numpy as np

def dag(array):
    return np.conj(np.swapaxes(array,-2,-1))

class OptOperation():

    def __init__(self):
        self.parameters = []
        self.optimization_on = False

    def forward(self):
        raise NotImplementedError("Please Implement this method")

    def grad_input(self, grad_output):
        raise NotImplementedError("Please Implement this method")

    def grad_parameters(self, grad_output):
        return []

    def backward(self, grad_output):
        grad_in = self.grad_input(grad_output)
        if self.optimization_on:
            grad_params = self.grad_parameters(grad_output)
        else:
            grad_params = []
        return grad_in, grad_params

    def update_parameters(self, params):
        pass


class OperationSequence(OptOperation):
    def __init__(self, sequence):       
        OptOperation.__init__(self)
        self.N_seq = len(sequence)
        self.sequence = sequence
    
    def forward(self, input):
        output = input
        for seq_ind in range(self.N_seq):
            output = self.sequence[seq_ind].forward(output)
        return output
    
    def grad_input(self, grad_output):
        grad_in = grad_output
        for seq_ind in range(self.N_seq):
            grad_in = self.sequence[seq_ind].grad_input(grad_output)
        return grad_in

    def grad_parameters(self, grad_output):
        grad_params = []
        for seq_ind in range(self.N_seq):
            grad_params += [self.sequence[seq_ind].grad_parameters(grad_output)]


class matmul(OptOperation):
    def __init__(self, mat):       
        OptOperation.__init__(self)
        self.parameters = mat
    
    def forward(self, input):
        self.input = input
        output = self.parameters @ input
        return output

    def grad_input(self, grad_output):
        grad_in = dag(self.parameters) @ grad_output
        return grad_in

    def grad_parameters(self, grad_output):
        grad_params = grad_output @ dag(self.input)
        return grad_params


class phase_diversity(OptOperation):
    def __init__(self, diversity):       
        OptOperation.__init__(self)
        self.parameters = diversity

    def forward(self, input):
        output = self.parameters[...,np.newaxis,np.newaxis] \
                 * input[...,np.newaxis,:,:]
        return output
    
    def grad_input(self, grad_output):
        grad_in = np.sum(grad_output 
            * np.conj(self.parameters[...,np.newaxis,np.newaxis]), axis=2)
        return grad_in


class polarization_diversity(OptOperation):
    def __init__(self, diversity):       
        OptOperation.__init__(self)
        self.parameters = diversity

    def forward(self, input):
        output = self.parameters @ input[...,np.newaxis,:,:]
        return output

    def grad_input(self, grad_output):
        grad_in = np.sum(dag(self.parameters) @ grad_output, axis=-2)
        return grad_in


class fft2(OptOperation):  
    def __init__(self):       
        OptOperation.__init__(self)
        
    def forward(self, input):   
        self.N_pts = input.shape[0]
        output = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(input, axes=(0,1)), axes=(0,1)),
            axes=(0,1)
            )/self.N_pts
        return output
    
    def grad_input(self, grad_output):
        grad_input = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(grad_output, axes=(0,1)), 
                         axes=(0,1)
                         ),
            axes=(0,1))*self.N_pts
        return grad_input


class scalmul():
    def __init__(self, scalar):       
        OptOperation.__init__(self)
        self.parameters = scalar

    def forward(self, input):
        self.input = input
        output = self.parameters * input
        return output

    def grad_input(self, grad_output):
        grad_in = self.parameters * grad_output
        return grad_in

    def grad_parameters(self, grad_output):
        grad_params = np.sum(grad_output * self.input, axis=(0,1))
        return grad_params


class scaladd():
    def __init__(self, scalar):       
        OptOperation.__init__(self)
        self.parameters = scalar

    def forward(self, input):
        self.input = input
        output = self.parameters + input
        return output

    def grad_input(self, grad_output):
        return grad_output

    def grad_parameters(self, grad_output):
        grad_params = np.sum(grad_output, axis=(0,1))
        return grad_params