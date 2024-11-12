import nltk
from nltk.corpus import wordnet

# Ensure that WordNet data is downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def get_synonym(word, pos_tag):
    # Fetch synsets of the word for the given part of speech
    synsets = wordnet.synsets(word, pos=pos_tag)
    # Check if any synsets are found, and return the first synonym if available
    if synsets:
        return synsets[0].lemmas()[0].name()
    return word

def replace_nouns(sentence):
    # Tokenize and POS-tag the sentence
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    
    # Replace nouns with synonyms
    new_sentence = []
    for word, pos in pos_tags:
        # NN (singular noun) or NNS (plural noun)
        if pos in ['NN', 'NNS']:
            new_word = get_synonym(word, wordnet.NOUN)
            new_sentence.append(new_word)
        else:
            new_sentence.append(word)
    
    return ' '.join(new_sentence)

# Original question
question = "A spacecraft with a mass of 841 kg is accelerating in space at a rate of 971,370 m/s². Using Newton's second law, calculate the force acting on the spacecraft."

# Replace nouns with synonyms
new_question = replace_nouns(question)
print(new_question)


