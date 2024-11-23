import nltk 
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize 
from nltk import pos_tag
nltk.download('averaged_perceptron_tagger_eng')

def is_noun(word):
    pos_tagged = pos_tag([word])
    tag = pos_tagged[0][1]
    noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
    return tag in noun_tags

def replace_words(sentence, units):
    words = word_tokenize(sentence)
    synsets = []
    for word in words:
        synset_list = wordnet.synsets(word)
        if synset_list and is_noun(word) and \
            (word not in units): synsets.append(synset_list[0].name().split('.')[0])
        else: synsets.append(word)
    return " ".join(synsets)

if __name__ ==  '__main__':
    sen = "The dog died and went to heaven."
    print(f"[INP]: {sen}")
    print(f"[OUT]: {replace_words(sen)}")