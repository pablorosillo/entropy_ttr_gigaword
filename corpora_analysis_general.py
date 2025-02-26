## One text with all the books aggregated

from nltk import FreqDist
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
import random
import numpy as np
from tqdm import tqdm
import pickle
import gzip
import pandas as pd
import re
import sys
import os
import ndd
import time

idxiniuser = int(sys.argv[1])
idxfinaluser = int(sys.argv[2])

init_time = time.time()

print('Libraries loaded')

def random_portion(tokens, num_tokens):

    num_tokens = int(num_tokens)
    
    # Asegura que inicio + num_tokens no sea mayor que la longitud del texto
    if len(tokens) >= num_tokens:
        # Genera un índice de inicio aleatorio
        inicio = random.randint(0, len(tokens) - num_tokens)
        
        # Obtiene num_tokens tokens a partir del índice de inicio
        tokens_seleccionados = tokens[inicio:inicio + num_tokens]
    
        return tokens_seleccionados

    else:
        print('El texto es demasiado corto para seleccionar', num_tokens, 'tokens')
        return None

def compute_type_to_token_ratio(tokens):
    types = set(tokens)
    
    token_count = len(tokens)
    type_count = len(types)
    
    if token_count == 0:
        return 0
    else:
        return  type_count/token_count

def compute_word_entropy(tokens):
    token_count = len(tokens)
    
    if token_count == 0:
        return 0
    else:
        freq_dist = FreqDist(tokens)
        probabilities = [freq_dist[token] / token_count for token in set(tokens)]
        counts = list(freq_dist.values())
        entropy_ml = -sum(p * np.log(p) for p in probabilities)
        entropy_nsb = ndd.entropy(counts, k=6*10**6)
        return entropy_ml, entropy_nsb

print('Functions defined')

##########################################################################################
##########################################################################################
#  Code for computing TTR and Entropy for random portions of the aggregated corpus
##########################################################################################
##########################################################################################

l_random_portions = np.arange(1000, 1800000000+2.5*10**6, 2.5*10**6)[idxiniuser:idxfinaluser]

# l_random_portions max value may change depending on the corpus

print(f'\nComputing from indexes {idxiniuser} to {idxfinaluser}')  
    
print(f'Random portions from {min(l_random_portions)} to {max(l_random_portions)} tokens will be selected')

# Load aggregated corpus

with gzip.open('/files/corpus_name.pkl', 'rb') as f:
    aggregated_corpus = pickle.load(f)

print('Aggregated corpus corpus_name.pkl loaded')

n_random_portions = 25

print('\nComputing TTR and Entropy for random portions of different sizes of the aggregated corpus')


columns=['L','Random TTR', 'Random TTR stdv', 'Random TTR Error', 'Random Entropy', 'Random Entropy stdv', 'Random Entropy Error',
         'Random NSB Entropy', 'Random NSB Entropy stdv', 'Random NSB Entropy Error']


df = pd.DataFrame(columns=columns)

for l in tqdm(l_random_portions):
    ttrs = []
    pi_entropies = []
    nsb_entropies = []

    for i in range(n_random_portions):

        random_portion_tokens = random_portion(aggregated_corpus, l)

        if random_portion_tokens is not None:

            ttrs.append(compute_type_to_token_ratio(random_portion_tokens))
            pi_entropy_iteration, nsb_entropy_iteration = compute_word_entropy(random_portion_tokens) 
            pi_entropies.append(pi_entropy_iteration)
            nsb_entropies.append(nsb_entropy_iteration)

    ttr_mean = np.mean(ttrs)
    ttr_stdv = np.std(ttrs)
    ttr_error = ttr_stdv / np.sqrt(n_random_portions)

    pi_entropy_mean = np.mean(pi_entropies)
    pi_entropy_stdv = np.std(pi_entropies)
    pi_entropy_error = pi_entropy_stdv / np.sqrt(n_random_portions)


    nsb_entropy_mean = np.mean(nsb_entropies)
    nsb_entropy_stdv = np.std(nsb_entropies)
    nsb_entropy_error = nsb_entropy_stdv / np.sqrt(n_random_portions)

    new_row = {
        'L': l,
        'Random TTR': ttr_mean,
        'Random TTR stdv': ttr_stdv,
        'Random TTR Error': ttr_error,
        'Random Entropy': pi_entropy_mean,
        'Random Entropy stdv': pi_entropy_stdv,
        'Random Entropy Error': pi_entropy_error,
        'Random NSB Entropy': nsb_entropy_mean,
        'Random NSB Entropy stdv': nsb_entropy_stdv,
        'Random NSB Entropy Error': nsb_entropy_error
    }
    
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)


print('Global dataframe created')

# Save dataframe using pickle and compress it with gzip

with gzip.open(f'files/dataframe_name.pkl', 'wb') as f:
    pickle.dump(df, f)

final_time = time.time()-init_time
print(f'Dataframe saved as /dataframe_name.pkl in {final_time} s.')
