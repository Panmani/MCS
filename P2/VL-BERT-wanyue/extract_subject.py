import spacy
from nltk.corpus import wordnet
nlp = spacy.load("en_core_web_sm")

with open('special_event.txt', 'r') as f:
    events = [line.rstrip() for line in f.readlines()]


# event = 'A group of teens all the way at the end of the hall are talking and messing around'
# person_list = []
# doc = nlp(event)
# noun_chunks = []
# for nc in list(doc.noun_chunks):
#     noun_chunks += str(nc).split(' ')
# for noun in noun_chunks:
#     syns = wordnet.synsets(noun)
#     if not syns:
#         continue
#     print(noun)
#     for syn in syns:
#         print(syn.name(), syn.lexname(), syn.definition())
#     print(' ')

# syns = wordnet.synsets('everyone')
# for syn in syns:
#     print(syn.name(), syn.lexname(), syn.definition())


i=0
for event in events:
    print(i)
    person_list = []
    doc = nlp(event)
    noun_chunks = []
    for nc in list(doc.noun_chunks):
        noun_chunks += str(nc).split(' ')

    for noun in set(noun_chunks):
        syns = wordnet.synsets(noun)
        if not syns:
            continue
        for i, syn in enumerate(syns):
            if syn.lexname() == "noun.person" or syn.lexname() == "noun.group":
                person_list.append((i, noun))
                break
    try:
        print('{} - [{}] - {}'.format(person_list[0], ','.join(person_list), event))
    except:
        print('[None] - [None] - {}'.format(event))

    i += 1
    if i > 20:
        break

