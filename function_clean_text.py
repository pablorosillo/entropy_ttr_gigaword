import re
from nltk.tokenize import word_tokenize

def textclean(text):

    # Removal of punctuation signs
    punct = r'[,;.:¡!¿?@#$%&[\](){}<>~=+\-*/|\\_^`"\']'
    text = re.sub(punct, ' ', text)

    # Removal of digits from 0 to 9 and lowercase
    text = re.sub('\d', ' ', text).lower()

    words = word_tokenize(text)

    # Remove ’ from the list

    words = [word for word in words if word != '’']

    return words