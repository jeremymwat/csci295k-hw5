import re
from collections import defaultdict
import nltk.data
import operator

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def basic_tokenizer(sentence, word_split=re.compile(b"([.,!?\"':;)(])")):
    """
    Very basic tokenizer: split the sentence into a list of tokens, lowercase.
    """
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(word_split, space_separated_fragment))
    return [w.lower() for w in words if w]

initial_word_list = []

with open("two_cities_hyphen.txt") as book:
  data = book.read().replace('\n', ' ')

sentence_list = tokenizer.tokenize(data.decode('utf-8'))

token_list = []

for sentence in sentence_list:
    token_list.extend(basic_tokenizer(sentence))
    token_list.append("<STOP>")

token_dict = defaultdict(int)
for token in token_list:
    token_dict[token] += 1

sorted_tokens = sorted(token_dict.items(), key=operator.itemgetter(1), reverse=True)

top_8000 = sorted_tokens[:8000]

just_words = set([tup[0] for tup in top_8000])

final_unked_list = []

for token in token_list:
    if token in just_words:
        final_unked_list.append(token)
    else:
        final_unked_list.append("*UNK*")

with open("tokenized_tails.txt", 'w') as book_out:
    book_out.write(' '.join(final_unked_list).encode('utf-8'))