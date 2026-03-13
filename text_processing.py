"""
1. Document/ text pricessing and embedding creation
"""

"""
to do list:
1. Import PDF file.
2. process text for embedding (e.g. split into chunk of sentences. )
3. Embed text chunk with embedding model.
4. save embeddings for later use.
"""

import os
import requests
import fitz
from tqdm.auto import tqdm
import pandas as pd
from spacy.lang.en import English
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'Learning to Optimize Tensor Programs.pdf')
filename = pdf_path

def text_formatter(text: str) -> str:
    import re
    # Replace newlines with spaces
    cleaned_text = text.replace('\n', ' ').strip()
    # Collapse multiple spaces
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    return cleaned_text

def open_and_read_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages_and_text = []
    for page_number, page in tqdm(enumerate(doc)):
        text = page.get_text()
        text = text_formatter(text=text)
        pages_and_text.append({"page_number" : page_number +1 ,
                                'page_char_count' : len(text),
                                'page_word_count' : len(text.split(' ')),
                                 'pages_sentence_count_raw' : len(text.split('. ')),
                                  'page_token_count' : len(text)/4,
                                   'text' : text })
    
    return pages_and_text

pages_and_text = open_and_read_pdf(filename)
#print(pages_and_text[0]) 


"""
next, we are going to embed the text into vector for model to process. 
since embedding model take finite tokens we need to split text into smaller chunk.
same for LLMs( context window take finite tokens).
"""
# group the sentences by pages
nlp = English()
nlp.add_pipe('sentencizer')

# Teach sentencizer not to split on common academic abbreviations
# by overriding sentence-start detection for tokens following these
ACADEMIC_ABBREVS = {
    'al', 'fig', 'Fig', 'eq', 'Eq', 'sec', 'Sec', 'ref', 'Ref',
    'approx', 'avg', 'max', 'min', 'vs', 'e.g', 'i.e', 'et', 'cf',
    'vol', 'no', 'pp', 'dr', 'prof', 'dept', 'univ',
}

def is_sentence_start(token):
    """Return False for tokens that follow academic abbreviations to avoid bad splits."""
    if token.is_sent_start and token.i > 0:
        prev = token.doc[token.i - 1]
        prev_text = prev.text.rstrip('.')
        if prev_text in ACADEMIC_ABBREVS:
            return False
    return token.is_sent_start

import re

def clean_sentences(sentences: list[str]) -> list[str]:
    """Filter out noise common in academic PDFs: headers, page numbers, short fragments."""
    cleaned = []
    for s in sentences:
        s = s.strip()
        # Skip very short fragments (likely headers, page numbers, figure labels)
        if len(s.split()) < 4:
            continue
        # Skip lines that are mostly numbers or references like "[1] Author..."
        if re.match(r'^\[?\d+\]?\.?\s', s):
            continue
        cleaned.append(s)
    return cleaned

for item in pages_and_text:
    item['sentences'] = list(nlp(item['text']).sents)
    item['sentences'] = [str(sentence) for sentence in item['sentences']]
    item['sentences'] = clean_sentences(item['sentences'])
    item['pages_sentence_count_spacy'] = len(item['sentences'])


# split list of sentences into a smaller chunk
num_sentence_chunk_size = 10
def split_list(input_list : list[str],
               slice_size : int = num_sentence_chunk_size) -> list[list[str]]:
    return [input_list[i : i + slice_size ] for i in range(0,len(input_list), slice_size)]

for item in pages_and_text:
    item['sentences_chunk'] = split_list(item['sentences'], num_sentence_chunk_size)

# splitting each chunk into its own item.
page_and_chunk = []
for item in pages_and_text:
    for sentence_chunk in item['sentences_chunk']:
        chunk_dict = {}
        chunk_dict['page_number'] = item['page_number']
        
        # join the sentences with a space (avoid corrupting punctuation)
        joined_sentences_chunk = ' '.join(sentence_chunk)

        chunk_dict['sentence_chunk'] = joined_sentences_chunk
        chunk_dict['chunk_char_count'] = len(joined_sentences_chunk)
        chunk_dict['chunk_word_count'] = len(joined_sentences_chunk.split())
        chunk_dict['chunk_token_count'] = len(joined_sentences_chunk) / 4  # ~4 chars per token
        page_and_chunk.append(chunk_dict)  # BUG FIX: was outside the inner loop



"""
embedding text chunk
"""
embedding_model = SentenceTransformer(model_name_or_path='all-mpnet-base-v2', device=device)

for item in tqdm(page_and_chunk):
    item['embedding'] = embedding_model.encode(item['sentence_chunk'])

"""
 for small dataset, we can store the embedding vector in csv file.
 for larger dataset,  we need a vectorbase for storage such as Qdrant, Chromadb , etc.
"""
text_chunk_and_embedding_df = pd.DataFrame(page_and_chunk)
text_chunk_and_embedding_df_save_path = 'text_chunk_and_embedding_df.csv'
text_chunk_and_embedding_df.to_csv(text_chunk_and_embedding_df_save_path, index= False)








