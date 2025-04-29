# tokenizer.py
def simple_tokenizer(text):
    word2idx = {'i':1, 'love':2, 'hate':3, 'this':4, 'movie':5, 'is':6, 'good':7, 'bad':8}
    tokens = [word2idx.get(word.lower(), 0) for word in text.split()]
    return tokens