# Advanced NLP with spaCy

### Create the `nlp` object
```python
from spacy.lang.en import English
nlp = English()
nlp = spacy.load("...")
```

### Types d'objet
- `spacy.Span` -> Collection de Tokens

### Tokens
- `token.pos_` -> pronom, verbe, nom
- `token.dep_` -> dépendance : sujet, ...
- `token.head.text` -> référence au token de `dep_`

Valeurs des `.dep_` :
- nsubj : nomimal subject
- dobj : direct object
- det : determiner

nlp("some text").ents -> itérable des entités  
ent.text, ent.label_  
spacy.explain(entity.label_ or "dobj" or any alias used in spacy)

### Matcher
`spacy.matcher` : kind of regex, using the grammar of the sentense  
=> ex : match a name, ignoring the case

```python
doc = nlp("hello world")
matcher = spacy.matcher.Matcher(nlp.vocab)  
matcher.add(name given, callback, patterns)
patterns = [
    {"ORTH": "iPhone"}, {"ORTH": "X"},
]  # tocken "iPhone" followed by tocken "X". Case sensitive

for match_id, idx_start, idx_end in matcher(doc):
    span = doc[idx_start: idx_end]
```

Other possibles patterns :
- "LEMMA" : the stem
- "LOWER" : ignore case
- "IS_PUNCT" : True / False
- "OP" : optionnal. "!" for absent, "?" for optionnal, + for 1+, * for 0+

Same with `matcher = spacy.matcher.PhraseMatcher(nlp.vocab)`

```python
doc = Doc(nlp.vocab, words=words, spaces=spaces)
span = Span(doc, idx_start, idx_end, label="label_name")
```

### Hashs
Spacy enregistre une seule fois les string, avec des hash
- Création d'un index hash -> word : `hash = nlp.vocab.strings[word]`
- et création d'un index word -> hash : `word = nlp.vocab.strings[hash]`

lexeme = nlp.vocab[word]
- word seulement, pas de contexte (nom ou verbe, précédent, ...)
- `lexeme.text`, `leveme.orth`, `lexeme.is_alpha`

### Semantic similarity:
- Require a model with words vectors included
- doc1.similarity(doc2)
- span1.similarity(span2)
- token1.similarity(token2)
- Generated using an algorithm like Words2Vec
- Default = cosine similarity

```python
doc1, doc2 = nlp(text1), nlp(text2)
doc1.similarity(doc2) -> Float in [0, 1]
```

**Vectors :**
```python
doc.vector
[token.vector fot token in doc]
```

### Pipeline
When running `doc = nlp(text)`, the steps are :
- Tokenizer
- Tagger : par of speech tagger :
  - `token.tag_`
- Parser : dependency parser :
  - `token.dep_`, `token.head`
  - `doc.sents`, `doc.noun_chunks`
- NER : named entity recognizer :
  - `token.ent_iop`, `token.ent_type`
  - `doc.ents`
- TextCat : text classifier :
  - `doc.cats`

Each model contains a file `meta.json`
- Define the language + name + pipeline steps (tagger, parser, ner)
- To display the pipeline steps : `nlp.pipe_names`, `nlp.pipeline`

Add a component in nlp pipeline :
- `nlp.add_pipe(function, first=True or last=True)`
- `nlp.add_pipe(lambda doc: ..., before="step name" or after="step name")`  
  Creates a new step named function.__name__

### Custom attributes
- Can be added in doc, span, token
- Access : `token._.attr_name`
- Can use an attribute, a property or a method to define a default value
- Attribute extension:
  - `Doc.set_extension('title', default=None)`
  - Same for Span and Token
- Property extension
  - `def get_is_color(token): return token.text in ["blue", "red"]`
  - `Token.set_extension("is_color", getter=get_is_color)`
- Method extension
  - `def has_token(doc, token_text): return token_text in [t.text for t in doc]`
  - `Doc.set_extension("has_token", method=has_token)`

### Optimisations :

**Process many texts :**  
`docs = [nlp(text) for text in lots_of_texts]`  
A remplacer par `docs = list(nlp.pipe(lots_of_texts))`

**Add context :**
```python
data = [("some text", {"page_num": 1},
        ("more text", {"page_num": 2}]
for doc, context in nlp.pipe(data, as_tuple=True:
    doc._.page_num = context["page_num"]
```

**Only process the tokenizer :**  
`doc = nlp("text")`  
A remplacer par `doc = nlp.make_doc("text")`

**Tempoary skip steps :**
```python
with nlp.disable_pipes("tagger", "parser"):
    doc = nlp("text")
```

### Training models
**Steps :**
1. Initialize the weights randomly with `nlp.begin_training`
1. Predict a few examples with `nlp.update`
1. Compare predictions with true labels
1. Calculate how to change weights
1. Update weights slightly
1. Go back to 2.

**Entity recognizer : text -> label**
- Don't forget to add text with no label (no entity) in the training
- To update a model : need minimum hundreds of samples
- To train a new category : a few thousand to millions samples
- Spacy length model : 2 million words

```python
entities = [(span.start_char, span.end_char, "LABEL")
            for span in ...]
training_example = (doc.text, {"entities": entities})
training_data = [training_example]
```

**Steps of training loop :**
- Loop for a number of times
- Shuffle the training data -> avoid sub-optimum
- Divide the data into batches (mini batching)
- Update the model for each batch
- Loop
- Save the model

```python
nlp = spacy.blank("en")
ner = nlp.create_pipe("ner")
nlp.add_pipe(ner)
ner.add_label("LABEL_NAME")
nlp.begin_training()

for i in range(N):
    losses = {}
    random.shuffle(training_data)
    
    for batch in spacy.util.minibatch(training_data):
        texts = [txt for txt, annotation in batch]
        annotations = [annotation for txt, annotation in batch]
        nlp.update(texts, annotations, losses=losses)
        print(losses)  # % d'erreur

nlp.to_disk(...)
```
% correct = (number of correct entities) / (number of expected entities)

### Training best practices
- Models can forget things :
  - If update data with new class : the model can forget one previous class
  - To train a new category, also include samples from the others (existing) categories
- Models can't learn everything
  - Don't use too specific labels

### Synthese
- Linguistic feature : part of speech tag, dependencies, named entity
- Pre-trained models
- Matcher, PhraseMatcher
- Doc, Span, Token, Vocab, Lexeme
- Find semantic similarities using word vectors
- Custom pipelines. Component with extension attribute
- Scale up SpaCy pipelines
- Create a dataset, train & update SpaCy's models

### Go further
- Training & updating other pipeline component :
  - part of speech tagger
  - dependency parser
  - text classifier
- Customizing the tokenizer
  - Add rules & exceptions to plit text differently
- Add / improve support for other languages
