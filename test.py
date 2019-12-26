import spacy

def word_embeddings(sequence):
    nlp = spacy.load("en_core_web_lg")
    vectors = {}
    sequence = nlp(sequence)
    print(sequence)
    for token in sequence:
        vectors[str(token)] = (token.vector)
    return vectors
    
x = word_embeddings("unk")
print(len(x.get("unk")))