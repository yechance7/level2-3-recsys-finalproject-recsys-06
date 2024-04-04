from gensim.models import FastText
import os

with open('corpus_total.txt', 'r') as f:
    corpus = f.read()


num_workers = os.cpu_count()-1
model = FastText(corpus_file='corpus_total.txt', vector_size=2000, window=20, min_count=5, workers=num_workers, sg=1, negative=100)

model.save('fasttext_vector2000_neg100.bin')