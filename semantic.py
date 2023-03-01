# Similarity with Spacy
import spacy
nlp = spacy.load('en_core_web_md')

word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))

# My note:
''' It's interesting how similarity works. Clearly, cats and monkeys are similar since they are both animals.
 Also, monkeys and bananas make a lot of sense as monkeys are known to eat bananas. 
 Finally, there are some small similarities between cats and bananas, but we still have a similarity 
 score of 0.22. I believe this percentage could be related to some cats being fed bananas'''


# My example:
word1 = nlp("food")
word2 = nlp("sun")
word3 = nlp("dog")

print(word1.similarity(word2))
print(word3.similarity(word2))
print(word3.similarity(word1))


# Working with vectors
nlp = spacy.load('en_core_web_dm')
tokens = nlp('cat apple monkey banana')

for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text, token1.similarity)

# Working with sentences
sentence_to_compare = "Why is my cat on the car"

sentences = ["where did my dog go",
             "Hello, there is my car",
             "I\ 've lost my car in my car",
             "I\ 'd like  my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
     similarity = nlp(sentence).similarity(model_sentence)
     print(sentence+ " - ", similarity)


''' My note 2:
The difference of using different models 'en_core_web_sm' and 'en_core_web_md' in this 
examples above is that produces slightly different similarity scores. This is because the 
models have been trained on different datasets, and therefore may have learned different 
word associations and language patterns. In general, the larger and more complex the model, 
the more accurate and nuanced the word embeddings are likely to be, and the more accurate 
the similarity scores will be. '''

