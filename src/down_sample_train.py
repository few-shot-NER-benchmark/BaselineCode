import os
import numpy as np

dataset = '../data/i2b2'
with open(os.path.join(dataset,'train.words')) as f, open(os.path.join(dataset,'train.ner')) as f_ner:
    with open(os.path.join(dataset,'train_10percent.words'),'w') as f_out, open(os.path.join(dataset,'train_10percent.ner'),'w') as f_out_ner:
        for line_words, line_ner in zip(f,f_ner):
            if np.random.uniform(0,1) < 0.1:
                f_out.write(line_words)
                f_out_ner.write(line_ner)
