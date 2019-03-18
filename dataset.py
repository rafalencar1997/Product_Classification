import os
import csv
import jsonlines
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset:

    def __init__(self, filename):
        self.filename = filename
        self.stopwords_en = set(stopwords.words('english'))
        self.regex_tokenizer = RegexpTokenizer(r'\w+')
        self.data = None
    
    @property
    def text(self):
        return self.data.text
    
    @property
    def label(self):
        return self.data.label
    
    @property
    def label_unique(self):
        return self.data.label.unique()
    
    def csv_to_json(self, columns_csv, columns_json):
        file_path = Path(self.filename)
        filename_jsonl = os.path.join(file_path.parent, file_path.stem + '.jsonl')

        readFile = open(self.filename, 'r')
        reader = csv.reader(readFile)

        data = list()
        for row in reader:
            data.append(row)
        readFile.close()

        dataset = pd.DataFrame.from_dict(data)
        dataset.columns = dataset.loc[0]
        dataset = dataset.drop([0])
        dataset = dataset.dropna()
        dataset = dataset[columns_csv]
        dataset.columns = columns_json
        dataset.to_json(filename_jsonl, orient='records', lines=True)

    def load(self, text_field='name', label_field='category', remove_stopwords=True, root_label=False):
        file_path = Path(self.filename)
        filename_jsonl = os.path.join(file_path.parent, file_path.stem + '.jsonl')
        
        dataset = list()
        not_labeled, not_text = 0, 0
        reader = jsonlines.open(filename_jsonl)

        for obj in reader:
            json = dict()
            if len(obj[label_field]) > 0:
                if len(obj[text_field]) > 0:
                    text = obj[text_field].lower()
                    if remove_stopwords:
                        tokens = self.regex_tokenizer.tokenize(text)
                        filtered_words = filter(lambda token: token not in self.stopwords_en, tokens)
                        text = " ".join(filtered_words)
                    if len(text) > 0:
                        json['text'] = text
                        if root_label:
                            json['label'] = obj[label_field].lower().split(' > ')[0]
                        else:
                            json['label'] = obj[label_field].lower()
                        dataset.append(json)
                    else:
                        not_text += 1  
                else:
                    not_text += 1    
            else:
                not_labeled += 1

        reader.close()
        print('%d lines skipped (not labeled)' % not_labeled)
        print('%d lines skipped (not text)' % not_text)
        self.data = pd.DataFrame.from_dict(dataset)
        
    def distrib_root_label(self, figsize=(15, 4)):
        labels = self.data.label.transform(lambda x: x.split(' > ')[0])
        labels_freq = labels.value_counts() / labels.count()
        labels_freq.plot(kind='bar', figsize=figsize)
        
    def distrib_label(self, figsize=(15, 80), min_samples=0):    
        group_data = self.label.value_counts()
        group_data = group_data[group_data >= min_samples]
        labels_freq = group_data / self.label.count()
        print('All labels with minimum %d samples: %d' % (min_samples, labels_freq.count()))
        labels_freq.plot(kind='barh', figsize=figsize, width=0.5)
        
    def filer_data(self):
        group_data = self.label.value_counts()
        group_data = group_data[group_data <= 1]
        self.data = self.data [~self.label.isin(group_data.index)]
        print('Total filtered labels:', self.label_unique.size)
    
    def max_text(self):
        return self.text.transform(lambda x: len(x.split())).max()
    
    def text_info(self):
        data = self.text.transform(lambda x: len(x.split()))

        maxL = data.max()
        minL = data.min()
        mode = data.mode()[0]
        mean = data.mean()
        std = data.std()
        s_w_ratio = self.data.text.count() / mean

        print('Text Length Max:  %d' % maxL)
        print('Text Length Min:  %d' % minL)
        print('Text Length Mode: %d' % mode)
        print('Text Length Mean: %.2f'% mean)
        print('Text Length Std:  %.2f'% std)
        print('S/W Text Ratio:   %.2f'% mean)

        plt.figure(figsize=(25,4))
        plt.hist(data, bins=maxL-1)
        
    def frequency_distribution_of_ngrams(self, ngram_range=(1, 2), num_ngrams=50):
        # Create args required for vectorizing.
        kwargs = {
                'ngram_range': (1, 1),
                'dtype': 'int32',
                'strip_accents': 'unicode',
                'decode_error': 'replace',
                'analyzer': 'word',  # Split text into word tokens.
        }
        vectorizer = CountVectorizer(**kwargs)
        vectorized_texts = vectorizer.fit_transform(self.data.text)

        # This is the list of all n-grams in the index order from the vocabulary.
        all_ngrams = list(vectorizer.get_feature_names())
        num_ngrams = min(num_ngrams, len(all_ngrams))
        # ngrams = all_ngrams[:num_ngrams]

        # Add up the counts per n-gram ie. column-wise
        all_counts = vectorized_texts.sum(axis=0).tolist()[0]

        # Sort n-grams and counts by frequency and get top `num_ngrams` ngrams.
        all_counts, all_ngrams = zip(*[(c, n) for c, n in sorted(
            zip(all_counts, all_ngrams), reverse=True)])
        ngrams = list(all_ngrams)[:num_ngrams]
        counts = list(all_counts)[:num_ngrams]

        idx = np.arange(num_ngrams)
        plt.figure(figsize=(20,5))
        plt.bar(idx, counts, width=0.8, color='b')
        plt.xlabel('N-grams')
        plt.ylabel('Frequencies')
        plt.title('Frequency distribution of n-grams')
        plt.xticks(idx, ngrams, rotation=45)
        
    def word_cloud(self, figsize=(20,10), max_words=150, mask=None):
        corpus = " ".join(text for text in self.text)
        # Create and generate a word cloud image:
        wordcloud = WordCloud(width=500, height=300, max_words=max_words, mask=mask,
                              background_color="white", repeat=False,
                              contour_width=3, contour_color='firebrick'
                             ).generate(corpus)
        # Display the generated image:
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        
    def tokenize(self, max_nb_words=20000):
        tokenizer = Tokenizer(num_words=max_nb_words)
        tokenizer.fit_on_texts(self.text)
        print('Number of Tokens:', len(tokenizer.word_counts))
        return tokenizer

    def vectorize(self, max_nb_words=20000):
        vectorizer  = TfidfVectorizer(max_features=max_nb_words)
        vectorizer.fit(self.text)
        return vectorizer

    def label_encode(self):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(self.label)
        return label_encoder
    
 