# DrumBeat.py
# author: Simon Shapiro
# date: 14 February 2017

import math
import random
from random import shuffle
from copy import deepcopy

class RTRL:
    def __init__(nn, numNeurons, neuronType, learningRate, pattern,\
    numSlots, runLength):
        '''
        numNeurons: The total number of neurons to be used.
        neuronType: The type of neuron to use, such as logsig, tansig, etc.
        learningRate: A real number indicating the learning rate for the network.
        pattern: A list of lists of slot numbers, with each list being a pattern for
        one percussion instrument. These instruments are played in parallel. Each number in a list should be between 0 and numSlots-1, indicating the slots within a cycle at which the corresponding instrument is to be hit. We do not encode the duration of the hit in the pattern.
        numSlots: The number of time steps in a repetition cycle.
        runLength: The total number of steps for which to run training and playing.
        '''
        nn.numNeurons = numNeurons
        nn.rate = learningRate
        nn.pattern = pattern
        nn.numSlots = numSlots
        
        # Input is an "index tape"
        nn.input = [0] * numSlots
        nn.input[0] = 1
        
        nn.runLength = runLength
        
        # First value is the value on the index tape, the second value is bias
        # which is always 1
        nn.x = [1,1]
        # External neurons
        nn.y = [0] * numNeurons
        # z = y + x
        nn.z = nn.x + nn.y
        
        nn.fprime = deepcopy(nn.y)
     
        nn.weights =[[randomWeight() for synapse in range(numNeurons+2)]
                                              for neuron in range(numNeurons)]
        nn.pkij=[[[0 for k in range(numNeurons+2)] 
                        for i in range(numNeurons)] 
                                for j in range(numNeurons)]
        nn.old_pkij = deepcopy(nn.pkij)
        
                                              
        nn.numOutputs = len(pattern)
        nn.errors = [0] * numNeurons
        
        nn.d = [[0] * numSlots] * nn.numOutputs
        for instrument in range(nn.numOutputs):
            for slot in pattern[instrument]:
                nn.d[instrument][slot] = 1
        
        nn.fun = neuronType.fun
        nn.deriv = neuronType.deriv
    
    def train(nn):

        allMSE = []
        
        for timestep in range(nn.runLength):
            # Will eventually populate this with the other paramaters asked for
            thisSlotOutput = [("Slot number", timestep)]
            
            currentBeat = timestep % nn.numSlots
            nn.x[0] = nn.input[currentBeat]
            nn.z = nn.y + nn.x
            
            # Copy over weights before modifying
            
            # Calculate new output
            nn.calcOutput()
            nn.z = nn.y + nn.x
            
            # Calculate the current error
            nn.calcError(currentBeat)

            # Calculate the new 3d matrix p_{kij}
            nn.calcPkij()
            
            for i in range(len(nn.weights)):
                for j in range(len(nn.weights[i])):                    
                    # Calculate the new_weight using Equation 15
                    delta_weight = 0
                    for k in range(len(nn.y)):
                        delta_weight += nn.errors[k] * nn.pkij[k][i][j]              
                    nn.weights[i][j] += nn.rate * delta_weight
            
                    
            
            
            # Calculate MSE
            # do you take the actual value from y, or do u take a 0 if < 0.5 and 1 if > 0.5
            MSE = inner(nn.errors, nn.errors) / nn.numOutputs
            allMSE.append(MSE)
            thisSlotOutput.append(("MSE", round(sum(allMSE)/(timestep+1), 3)))
            
            # Add click track value to output
            thisSlotOutput.append(("Click track value", nn.x[0]))
            
            # Training values Vs. output values
            for i in range(nn.numOutputs):
                if nn.y[i] < 0.5:
                    output_value = 0
                else: output_value = 1
                thisSlotOutput.append(("Instrument " + str(i) + " training " + \
                "value vs. output value", output_value, nn.d[i][currentBeat]))
                
            print thisSlotOutput
            '''
            print "d:", nn.d
            print "y:", [round(y,3) for y in nn.y]
            print "errors:", [round(e,3) for e in nn.errors]
            '''
            #timeline.append(thisSlotOutput)
            

        print "d:", nn.d
        print "y:", [round(y,3) for y in nn.y]
        print "errors:", [round(e,3) for e in nn.errors]
                    
    
    def calcError(nn, currentBeat):
        '''
        Calculate the error using Equation 4.
        Only update the error values for the instrument neurons. The rest are always
        0.
        '''
        for instrument in range(nn.numOutputs):
            nn.errors[instrument] = \
            nn.d[instrument][currentBeat] - nn.y[instrument]
            
    def calcOutput(nn):
        ''' 
        Calculate output and deriv to prepare for calcPkij
        '''
        # Equation 2
        s_k = [inner(weight,nn.z) for weight in nn.weights]
        
        #nn.next_output = [nn.fun(act) for act in s_k]
        nn.y = [nn.fun(act) for act in s_k]

        '''
        print "weights:", nn.weights
        print "nn.z:", nn.z
        print "s_k:", s_k
        print "y:", [round(y,3) for y in nn.y]
        '''
        
        
        for neuron in range(len(nn.y)):
            nn.fprime[neuron] = nn.deriv(s_k[neuron], nn.y[neuron])
    
    def calcPkij(nn):
        '''
        Calculate the values for the 3d matrix p using equation 12
        '''
        nn.old_pkij = deepcopy(nn.pkij)
        for k in range(len(nn.pkij)):
            for i in range(len(nn.pkij[k])):
                for j in range(len(nn.pkij[k][i])):
                    # Calculate sum term from eq. 12
                    wp = 0
                    for l in range(len(nn.y)):
                        wp += nn.weights[k][l] * nn.old_pkij[l][i][j]
                    # Kronecker term
                    if i == k:
                        wp += nn.z[j]
                    nn.pkij[k][i][j] = nn.fprime[k] * wp
    
    
            
                    
class ActivationFunction:
    """ ActivationFunction packages a function together with its derivative. """
    """ This prevents getting the wrong derivative for a given function.     """
    """ Because some derivatives are computable from the function's value,   """
    """ the derivative has two arguments: one for the argument and one for   """
    """ the value of the corresponding function. Typically only one is use.  """

    def __init__(af, name, fun, deriv):
        af.name = name
        af.fun = fun
        af.deriv = deriv

    def fun(af, x):
        return af.fun(x)

    def deriv(af, x, y):
        return af.deriv(x, y)

logsig = ActivationFunction("logsig",
                            lambda x: 1.0/(1.0 + math.exp(-x)),
                            lambda x,y: y*(1.0-y))

tansig = ActivationFunction("tansig",
                            lambda x: math.tanh(x),
                            lambda x,y: 1.0 - y*y)

purelin = ActivationFunction("purelin",
                             lambda x: x,
                             lambda x,y: 1)

def randomWeight():
    """ returns a random weight value between -0.5 and 0.5 """
    return random.random()-0.5

def inner(x, y):
    """ Returns the inner product of two equal-length vectors. """
    n = len(x)
    assert len(y) == n
    sum = 0
    for i in range(0, n):
        sum += x[i]*y[i]
    return sum

def kick():
    nn = RTRL(16, logsig, 0.1, [[0,8,10]], 16, 10000)
    nn.train()

def basic():
    nn = RTRL(4, logsig, 0.1, [[0,1]], 4, 10000)
    nn.train()

def basic2():
    nn = RTRL(4, logsig, 0.1, [[1,3]], 4, 9999)
    nn.train()

hiphopSamples = [[7], [0,1,2,3,4,5,6,8,9,10,11,12,13,14], [4, 12], [0, 8, 10]]

def hiphop():
    nn = RTRL(4 * 16, logsig, 0.1, hiphopSamples, 16, 10000)
    nn = RTRL()
    
def main():
    #basic()
    basic2()
     #kick()
    # hiphop()
    #
main()