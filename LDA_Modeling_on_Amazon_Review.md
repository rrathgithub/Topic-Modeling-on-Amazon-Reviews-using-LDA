

```pyspark3
sc
```

    Starting Spark application



<table>
<tr><th>ID</th><th>YARN Application ID</th><th>Kind</th><th>State</th><th>Spark UI</th><th>Driver log</th><th>Current session?</th></tr><tr><td>0</td><td>application_1544400356427_0001</td><td>pyspark3</td><td>idle</td><td><a target="_blank" href="http://ip-172-31-44-76.ec2.internal:20888/proxy/application_1544400356427_0001/">Link</a></td><td><a target="_blank" href="http://ip-172-31-37-187.ec2.internal:8042/node/containerlogs/container_1544400356427_0001_01_000001/livy">Link</a></td><td>✔</td></tr></table>


    SparkSession available as 'spark'.
    <SparkContext master=yarn appName=livy-session-0>

## Summary:
- Access the Amazon review dataset from "s3://amazon-reviews-pds/parquet" and select a subset from headphone category;
- Process the data using lemmatization, tagging and tokenization;
- Group the reviews based on rating scores;
- Train LDA model and test the model.

### Data loading


```pyspark3
reviews = spark.read.parquet("s3://amazon-reviews-pds/parquet")
```


```pyspark3
reviews_elec = reviews.where("product_category = 'Electronics' \
                             and marketplace = 'US' \
                             and lower(product_title) like '%bluetooth%' \
                             and (product_title like '%earphone%' \
                             or product_title like '%headphone%') ")
```


```pyspark3
df.printSchema()
```

    root
     |-- marketplace: string (nullable = true)
     |-- customer_id: string (nullable = true)
     |-- review_id: string (nullable = true)
     |-- product_id: string (nullable = true)
     |-- product_parent: string (nullable = true)
     |-- product_title: string (nullable = true)
     |-- star_rating: integer (nullable = true)
     |-- helpful_votes: integer (nullable = true)
     |-- total_votes: integer (nullable = true)
     |-- vine: string (nullable = true)
     |-- verified_purchase: string (nullable = true)
     |-- review_headline: string (nullable = true)
     |-- review_body: string (nullable = true)
     |-- review_date: date (nullable = true)
     |-- year: integer (nullable = true)
     |-- product_category: string (nullable = true)


```pyspark3
df = reviews_elec
```

### Data Pre-processing


```pyspark3
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.functions import collect_list

from nltk import download
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import RegexpParser
from nltk import pos_tag

import string
import re
import itertools
```

#### Remove non ASCII characters


```pyspark3
# remove non ASCII characters
def strip_non_ascii(data_str):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in data_str if 0 < ord(c) < 127)
    return ''.join(stripped)

# setup pyspark udf function
strip_non_ascii_udf = udf(strip_non_ascii, StringType())
```


```pyspark3
# applying the user defined function of removing non ASCII characters
df = df.withColumn('text_non_asci',strip_non_ascii_udf(df['review_body']))
```

#### Fixed abbreviation


```pyspark3
# modify abbreviations
def fix_abbreviation(data_str):
    data_str = data_str.lower()
    data_str = re.sub(r'\bthats\b', 'that is', data_str)
    data_str = re.sub(r'\bive\b', 'i have', data_str)
    data_str = re.sub(r'\bim\b', 'i am', data_str)
    data_str = re.sub(r'\bya\b', 'yeah', data_str)
    data_str = re.sub(r'\bcant\b', 'can not', data_str)
    data_str = re.sub(r'\bdont\b', 'do not', data_str)
    data_str = re.sub(r'\bwont\b', 'will not', data_str)
    data_str = re.sub(r'\bid\b', 'i would', data_str)
    data_str = re.sub(r'wtf', 'what the fuck', data_str)
    data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
    data_str = re.sub(r'\br\b', 'are', data_str)
    data_str = re.sub(r'\bu\b', 'you', data_str)
    data_str = re.sub(r'\bk\b', 'OK', data_str)
    data_str = re.sub(r'\bsux\b', 'sucks', data_str)
    data_str = re.sub(r'\bno+\b', 'no', data_str)
    data_str = re.sub(r'\bcoo+\b', 'cool', data_str)
    data_str = re.sub(r'rt\b', '', data_str)
    data_str = data_str.strip()
    return data_str

# setup pyspark udf function
fix_abbreviation_udf = udf(fix_abbreviation, StringType())
```


```pyspark3
# applying the user defined function of modifying abbreviations
df = df.withColumn('text_fixed_abbrev',fix_abbreviation_udf(df['text_non_asci']))
```

#### Remove hyperlinks, puncuations, numbers, etc.


```pyspark3
# remove hyperlinks, puncuations, numbers, etc.
def remove_features(data_str):
    # compile regex
    url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
    punc_re = re.compile('[%s]' % re.escape(string.punctuation))
    num_re = re.compile('(\\d+)')
    mention_re = re.compile('@(\w+)')
    alpha_num_re = re.compile("^[a-z0-9_.]+$")
    html_re = re.compile("<br />")
    # convert to lowercase
    data_str = data_str.lower()
    # remove hyperlinks
    data_str = url_re.sub(' ', data_str)
    # remove @mentions
    data_str = mention_re.sub(' ', data_str)
    # remove puncuation
    data_str = punc_re.sub(' ', data_str)
    # remove numeric 'words'
    data_str = num_re.sub(' ', data_str)
    # remove html symbol
    data_str = html_re.sub(' ', data_str)   
    # remove non a-z 0-9 characters and words shorter than 1 characters
    list_pos = 0
    cleaned_str = ''
    for word in data_str.split():
        if list_pos == 0:
            if alpha_num_re.match(word) and len(word) > 1:
                cleaned_str = word
            else:
                cleaned_str = ' '
        else:
            if alpha_num_re.match(word) and len(word) > 1:
                cleaned_str = cleaned_str + ' ' + word
            else:
                cleaned_str += ' '
        list_pos += 1
    # remove unwanted space, *.split() will automatically split on
    # whitespace and discard duplicates, the " ".join() joins the
    # resulting list into one string.
    return " ".join(cleaned_str.split())

# setup pyspark udf function
remove_features_udf = udf(remove_features, StringType())
```


```pyspark3
# applying the user defined function of removing hyperlinks, punctuations, numbers, etc.
df = df.withColumn('text_feature_removed',remove_features_udf(df['text_fixed_abbrev']))
```

#### Group together the different inflected forms of a word 

- convert past tense and future tense into simple present tense
- convert plural form into singular form


```pyspark3
# filter out the empty non-type values
df = df.where(df.text_feature_removed.isNotNull())
```


```pyspark3
# Group together the different inflected forms of a word
def lemmatize(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = pos_tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

# setup pyspark udf function
lemmatize_udf = udf(lemmatize, StringType())
```


```pyspark3
# applying the user defined function of lemmatizing words with different tenses and forms
lemm_df = df.withColumn("lemm_text", lemmatize_udf(df["text_feature_removed"]))
```

#### Mark up a word in a text as corresponding to a particular part of speech, based on both its definition and its context 

- Identify different part of the speech
- Combine patterns such as "noun + noun" and "adjective + noun"


```pyspark3
# filter out the empty non-type values
lemm_df = lemm_df.where(lemm_df.lemm_text.isNotNull())
```


```pyspark3
def tag_and_remove(data_str):
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP','NNS','NNP','NNPS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags

    # break string into 'words'
    text = data_str.split()
    
    text_notype = []
    for w in text:
        if w is None:
            continue
        else:
            text_notype.append(w)

    # tag the text and keep only those with the right tags
    tagged_text = pos_tag(text_notype)
    
    for i in range(len(tagged_text)):
        if tagged_text[i][1] in nltk_tags:
            if i < len(tagged_text)-1:
                if (tagged_text[i][1] in nn_tags) and (tagged_text[i+1][1] in nn_tags):
                    cleaned_str += tagged_text[i][0] + '_'
                elif (tagged_text[i][1] in jj_tags) and (tagged_text[i+1][1] in nn_tags):
                    cleaned_str += tagged_text[i][0] + '_'  
                else:
                    cleaned_str += tagged_text[i][0] + ' '
#    for tagged_word in tagged_text:
#        if tagged_word[1] in nltk_tags:
#            cleaned_str += tagged_word[0] + ' '
            

    return cleaned_str

# setup pyspark udf function
tag_and_remove_udf = udf(tag_and_remove, StringType())
```


```pyspark3
# applying the user defined function of tagging by part of speech
tagged_df = lemm_df.withColumn("tag_text", tag_and_remove_udf(lemm_df.lemm_text))
```

#### Remove stopwords


```pyspark3
# filter out the empty non-type values
tagged_df = tagged_df.where(tagged_df.tag_text.isNotNull())
```


```pyspark3
from nltk.corpus import stopwords
download('stopwords')
stop_words = stopwords.words('english')
stop_words.append('br')
stop_words.append('would')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /var/lib/livy/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.


```pyspark3
# remove stop words
def remove_stops(data_str):
    # expects a string
    #stops = set(stopwords.words("english"))
    list_pos = 0
    cleaned_str = ''
    text = data_str.split()
    for word in text:
        if word not in stop_words:
            # rebuild cleaned_str
            if list_pos == 0:
                cleaned_str = word
            else:
                cleaned_str = cleaned_str + ' ' + word
            list_pos += 1
    return cleaned_str

# setup pyspark udf function
remove_stops_udf = udf(remove_stops, StringType())
```


```pyspark3
# applying the user defined function of removing stop words
stop_df= tagged_df.withColumn("stop_text", remove_stops_udf(tagged_df["tag_text"]))
```

#### Tokenize the reviews into words


```pyspark3
# setup pyspark udf function
tokenize_udf = udf(word_tokenize, ArrayType(StringType()))

token_df = stop_df.withColumn("token_text", tokenize_udf(stop_df["stop_text"]))
```

#### Set the reviews from 2015 as testing dataset and others as training datasets
#### Group reviews by ratings and years


```pyspark3
from pyspark.sql.functions import collect_list
```


```pyspark3
# filter out the empty non-type values
token_df = token_df.where(token_df.token_text.isNotNull())
```


```pyspark3
df_combine_train = token_df.where("year != 2015").groupby('star_rating').agg(collect_list('token_text').alias("review_clean"))
```


```pyspark3
df_combine_test = token_df.where("year = 2015").groupby('star_rating').agg(collect_list('token_text').alias("review_clean"))
```


```pyspark3
df_combine_train = df_combine_train.where(df_combine_train.review_clean.isNotNull())
df_combine_test = df_combine_test.where(df_combine_test.review_clean.isNotNull())
```


```pyspark3
import itertools
```


```pyspark3
# Flatten the nested lists
def flatten_nested_list(nested_list):
    flatten_list = list(itertools.chain.from_iterable(nested_list))
    return flatten_list
```


```pyspark3
flatten_udf = udf(flatten_nested_list, ArrayType(StringType()))
```


```pyspark3
df_combine_train = df_combine_train.withColumn('review_cleaned', flatten_udf(df_combine_train.review_clean))
df_combine_test = df_combine_test.withColumn('review_cleaned', flatten_udf(df_combine_test.review_clean))
```


```pyspark3
texts_train = df_combine_train.sort("star_rating",ascending=True).select('star_rating','review_cleaned').collect()
texts_test = df_combine_test.sort("star_rating",ascending=True).select('star_rating','review_cleaned').collect()
```

#### Create the documents for LDA


```pyspark3
documents_train = []
for i in range(len(texts_train)):
    documents_train.append(texts_train[i].review_cleaned)
```


```pyspark3
documents_test = []
for i in range(len(texts_test)):
    documents_test.append(texts_test[i].review_cleaned)
```

#### Filtering out the most frequent words


```pyspark3
dict_train = {}
for i in documents_train:
    for j in i:
        if j in dict_train.keys():
            dict_train[j] += 1
        else:
            dict_train[j] = dict_train.get(j, 0) + 1
```


```pyspark3
dict_test = {}
for i in documents_test:
    for j in i:
        if j in dict_test.keys():
            dict_test[j] += 1
        else:
            dict_test[j] = dict_test.get(j, 0) + 1
```


```pyspark3
n_frequent_words_train=[]
for k, v in dict_train.items():
    if v > 90:
        n_frequent_words_train.append(k)
```


```pyspark3
n_frequent_words_test=[]
for k, v in dict_test.items():
    if v > 75:
        n_frequent_words_test.append(k)
```


```pyspark3
documents_train_filter = []
for l in documents_train:
    new_l = l
    for w in new_l:
        if w in n_frequent_words_train:
            new_l.remove(w)
    documents_train_filter.append(new_l)
```


```pyspark3
documents_test_filter = []
for l in documents_test:
    new_l = l
    for w in new_l:
        if w in n_frequent_words_test:
            new_l.remove(w)
    documents_test_filter.append(new_l)
```

### Run the LDA model


```pyspark3
from gensim import corpora, models
```

#### Create the dictionary for documents

The 'Dictionary()' fucntion answers the question -- how many times does a specific word appear in the document? 
The function would assign a unique integer id to all words appearing in the corpus. 


```pyspark3
dictionary_train_1 = corpora.Dictionary([documents_train_filter[0]])
dictionary_train_2 = corpora.Dictionary([documents_train_filter[1]])
dictionary_train_3 = corpora.Dictionary([documents_train_filter[2]])
dictionary_train_4 = corpora.Dictionary([documents_train_filter[3]])
dictionary_train_5 = corpora.Dictionary([documents_train_filter[4]])
```

#### Convert tokenized documents to vectors

The 'doc2bow()' function counts the number of occurrences of each distinct word, converts the word to its integer word id and returns the result as a sparse vector.


```pyspark3
bow_corpus_train_1 = [dictionary_train_1.doc2bow(doc) for doc in [documents_train_filter[0]]]
bow_corpus_train_2 = [dictionary_train_2.doc2bow(doc) for doc in [documents_train_filter[1]]]
bow_corpus_train_3 = [dictionary_train_3.doc2bow(doc) for doc in [documents_train_filter[2]]]
bow_corpus_train_4 = [dictionary_train_4.doc2bow(doc) for doc in [documents_train_filter[3]]]
bow_corpus_train_5 = [dictionary_train_5.doc2bow(doc) for doc in [documents_train_filter[4]]]
```

#### Train the LDA model

id2word: The model requires the previous dictionary to map ids to strings.
passes: The number of laps the model will take through corpus. The greater the number of passes, the more accurate the model will be.


```pyspark3
from gensim.models import LdaModel
```


```pyspark3
lda_model_1 = LdaModel(bow_corpus_train_1, num_topics=3, id2word=dictionary_train_1, passes=1)
lda_model_2 = LdaModel(bow_corpus_train_2, num_topics=3, id2word=dictionary_train_2, passes=1)
lda_model_3 = LdaModel(bow_corpus_train_3, num_topics=3, id2word=dictionary_train_3, passes=1)
lda_model_4 = LdaModel(bow_corpus_train_4, num_topics=3, id2word=dictionary_train_4, passes=1)
lda_model_5 = LdaModel(bow_corpus_train_5, num_topics=3, id2word=dictionary_train_5, passes=1)
```

#### Print the topics within each model


```pyspark3
for idx, topic in lda_model_1.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.008*"try" + 0.008*"phone" + 0.007*"put" + 0.007*"hear" + 0.007*"look" + 0.007*"product" + 0.006*"money" + 0.005*"return" + 0.005*"break" + 0.005*"purchase"
    Topic: 1 
    Words: 0.010*"phone" + 0.009*"return" + 0.009*"try" + 0.008*"look" + 0.008*"hear" + 0.007*"break" + 0.007*"music" + 0.007*"come" + 0.007*"put" + 0.007*"money"
    Topic: 2 
    Words: 0.009*"phone" + 0.009*"return" + 0.008*"try" + 0.008*"put" + 0.007*"hear" + 0.007*"money" + 0.007*"product" + 0.007*"music" + 0.006*"fit" + 0.006*"sound_quality"


```pyspark3
for idx, topic in lda_model_2.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.007*"cable" + 0.006*"find" + 0.005*"need" + 0.005*"love" + 0.005*"come" + 0.005*"pair" + 0.005*"fit" + 0.004*"look" + 0.004*"comfortable" + 0.004*"perfect"
    Topic: 1 
    Words: 0.007*"love" + 0.006*"bose" + 0.006*"fit" + 0.005*"cable" + 0.005*"come" + 0.005*"comfortable" + 0.005*"need" + 0.005*"music" + 0.004*"find" + 0.004*"listen"
    Topic: 2 
    Words: 0.007*"love" + 0.007*"come" + 0.006*"need" + 0.006*"find" + 0.006*"cable" + 0.005*"fit" + 0.005*"bose" + 0.005*"recommend" + 0.005*"price" + 0.005*"expect"


```pyspark3
for idx, topic in lda_model_3.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.009*"run" + 0.007*"think" + 0.007*"give" + 0.006*"fall" + 0.006*"stay" + 0.005*"know" + 0.005*"charge" + 0.005*"keep" + 0.005*"something" + 0.005*"say"
    Topic: 1 
    Words: 0.008*"stay" + 0.008*"think" + 0.008*"come" + 0.007*"run" + 0.007*"keep" + 0.006*"say" + 0.006*"know" + 0.006*"item" + 0.006*"something" + 0.005*"charge"
    Topic: 2 
    Words: 0.009*"run" + 0.008*"keep" + 0.008*"come" + 0.006*"fall" + 0.006*"stay" + 0.006*"say" + 0.006*"know" + 0.006*"try" + 0.005*"think" + 0.005*"ear_bud"


```pyspark3
for idx, topic in lda_model_4.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.005*"bass" + 0.005*"music" + 0.004*"say" + 0.004*"pair" + 0.004*"give" + 0.004*"come" + 0.004*"hear" + 0.004*"seem" + 0.003*"wear" + 0.003*"look"
    Topic: 1 
    Words: 0.005*"pair" + 0.005*"bass" + 0.004*"time" + 0.004*"sound_quality" + 0.004*"music" + 0.004*"fit" + 0.004*"want" + 0.004*"purchase" + 0.004*"give" + 0.004*"earbuds"
    Topic: 2 
    Words: 0.005*"pair" + 0.005*"music" + 0.004*"run" + 0.004*"sound_quality" + 0.004*"give" + 0.004*"bass" + 0.004*"keep" + 0.004*"say" + 0.004*"earbuds" + 0.004*"purchase"


```pyspark3
for idx, topic in lda_model_5.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))
```

    Topic: 0 
    Words: 0.012*"fit" + 0.006*"say" + 0.006*"want" + 0.006*"try" + 0.005*"give" + 0.005*"head" + 0.005*"little" + 0.005*"take" + 0.005*"loud" + 0.005*"hear"
    Topic: 1 
    Words: 0.011*"fit" + 0.006*"say" + 0.006*"something" + 0.005*"try" + 0.005*"price" + 0.005*"pair" + 0.005*"loud" + 0.005*"want" + 0.005*"head" + 0.004*"give"
    Topic: 2 
    Words: 0.010*"fit" + 0.006*"say" + 0.006*"give" + 0.005*"come" + 0.005*"take" + 0.004*"little" + 0.004*"want" + 0.004*"time" + 0.004*"loud" + 0.004*"something"

#### Test the model using reviews from 2015


```pyspark3
bow_vector_1 = dictionary_train_1.doc2bow(documents_test_filter[0])
bow_vector_2 = dictionary_train_2.doc2bow(documents_test_filter[4])
bow_vector_3 = dictionary_train_3.doc2bow(documents_test_filter[1])
bow_vector_4 = dictionary_train_4.doc2bow(documents_test_filter[3])
bow_vector_5 = dictionary_train_5.doc2bow(documents_test_filter[2])
```


```pyspark3
for index, score in sorted(lda_model_1[bow_vector_1], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_1.print_topic(index, 5)))
```

    Score: 0.7029314041137695	 Topic: 0.010*"phone" + 0.009*"return" + 0.009*"try" + 0.008*"look" + 0.008*"hear"
    Score: 0.2963051199913025	 Topic: 0.009*"phone" + 0.009*"return" + 0.008*"try" + 0.008*"put" + 0.007*"hear"


```pyspark3
for index, score in sorted(lda_model_2[bow_vector_2], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_2.print_topic(index, 5)))
```

    Score: 0.9995627403259277	 Topic: 0.007*"love" + 0.007*"come" + 0.006*"need" + 0.006*"find" + 0.006*"cable"


```pyspark3
for index, score in sorted(lda_model_3[bow_vector_3], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_3.print_topic(index, 5)))
```

    Score: 0.9527735114097595	 Topic: 0.009*"run" + 0.008*"keep" + 0.008*"come" + 0.006*"fall" + 0.006*"stay"
    Score: 0.046444643288850784	 Topic: 0.008*"stay" + 0.008*"think" + 0.008*"come" + 0.007*"run" + 0.007*"keep"


```pyspark3
for index, score in sorted(lda_model_4[bow_vector_4], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_4.print_topic(index, 5)))
```

    Score: 0.9954943656921387	 Topic: 0.005*"pair" + 0.005*"bass" + 0.004*"time" + 0.004*"sound_quality" + 0.004*"music"


```pyspark3
for index, score in sorted(lda_model_5[bow_vector_5], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model_5.print_topic(index, 5)))
```

    Score: 0.5268938541412354	 Topic: 0.012*"fit" + 0.006*"say" + 0.006*"want" + 0.006*"try" + 0.005*"give"
    Score: 0.47252732515335083	 Topic: 0.011*"fit" + 0.006*"say" + 0.006*"something" + 0.005*"try" + 0.005*"price"

### Results

After training five different models based on ratings, when a new comment is published by customer, we can infer the topics within that comment by processing it with current model.
In that case, if the score is low, which means the new comment has a totally different topic, we should be cautious because it is possible that the customers' preference has changed or some new issues have appeared about the product.
