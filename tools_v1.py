import random
import numpy as np

def shuffle(lol, seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def padding(sentence, length, pad):
    num_pads = length - len(sentence)
    return np.pad(sentence,(0,num_pads), 'constant', constant_values=(pad))

def batching(sentence, batch_size, pad):
    end_batch = len(sentence)%batch_size
    if end_batch>0:
        remaining = batch_size-end_batch
        new_sentence = padding(sentence,len(sentence)+remaining,pad)
    else:
        new_sentence = sentence
    return [new_sentence[i:i+batch_size] for i in range(0,len(new_sentence),batch_size)]
