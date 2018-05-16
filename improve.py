from sklearn.linear_model import LogisticRegression
from sklearn import tree

from nltk.corpus import reuters, wordnet
from sklearn.datasets import fetch_20newsgroups

import spacy
import pyphen

from collections import Counter
import os, re

class Improved(object):

    def __init__(self, language):
        self.language = language
    
        if language == 'english':

            # Using Reference Corpus - Reuters, WordNet, Wikipedia, 20News
            # Uncomment to include Wiki & 20News
            self.corpusReuters = Counter(reuters.words())
            self.corpusWordNet = Counter(wordnet.words())
            
            # 20News Corpus
#            words20News = {}
#            self.corpus20News = fetch_20newsgroups (remove = ('headers', 'footers', 'quotes'))
#            
#            for sent in self.corpus20News['data']:
#                words20News.update (Counter(re.findall("[A-Za-z]+", sent.lower() )))
#           
#            self.words20News = words20News
#            
#            
#            # Wiki Corpus
#            wordsWikiEn = {}            
#            for docs in os.listdir("Corpus/raw.en/"):
#                with open ("Corpus/raw.en/" + docs, 'r', encoding = 'utf-8', errors='ignore') as file:
#                    for line in file:
#                        wordsWikiEn.update (Counter(re.findall("[A-Za-z]+", line.lower())))
#
#            self.wordsWikiEn = wordsWikiEn
#         
#            
#        else:  # spanish
#            wordsWikiEs = {}            
#            for docs in os.listdir("Corpus/raw.es/"):
#                with open ("Corpus/raw.es/" + docs, 'r', encoding = 'utf-8', errors='ignore') as file:
#                    for line in file:
#                        wordsWikiEs.update (Counter(re.findall("[A-Za-z]+", line.lower())))
#
#            self.wordsWikiEs = wordsWikiEs
                       
            
        # Model - Logistic Regression & Decision Tree Classifier
        self.model1 = LogisticRegression()
        self.model2 = tree.DecisionTreeClassifier()
        

    # Frequency of Words based on Reference Corpus
    # English (Reuters, WordNet, 20News, Wikipedia) - Spanish (Wikipedia)
    def extract_features1 (self, word, language):
        
        if language == 'english':
            
            # Frequency Count for each corpus
            countReuters = self.corpusReuters[word]
            countWordNet = self.corpusWordNet[word]
            
            count20News = self.words20News.get(word)
            if count20News == 'Inf' or 'NaN':
                count20News = 0
          
            countWiki = self.wordsWikiEn.get(word)
            if countWiki == 'Inf' or 'Nan':
                countWiki = 0
            
            return [countReuters, countWordNet, count20News, countWiki]
        
        else: # spanish
           countWiki = self.wordsWikiEs.get(word)
           if countWiki == 'Inf' or 'Nan':
                countWiki = 0
        
           return [countWiki]


    # Length of words (without averaging)
    def extract_features2 (self, word, language):
        
        lenWords = len(word)
        lenTokens = len(word.split(' '))

        return [lenTokens, lenWords]
    
    
    # Detecting if the word is NER or not - not suitable for Logistic Regression
    def extract_features3 (self, word, language):
        
        if language == 'english':
            nlpModel = spacy.load('en_core_web_sm', disable = ['parser'])
            
        else: # Spanish
            nlpModel = spacy.load('es_core_news_sm', disable = ['parser'])
        
        NERtester = nlpModel(word, disable = ['parser'])
                
        if NERtester[0].ent_type_ == ' ':
            countNER = 1
        else:
            countNER = 0
                
        return [countNER]
                
    
    # Syllables Counts
    # Ref - https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word
    # Ref - https://www.thoughtco.com/what-is-complex-word-1689889
    # Ref - http://pyphen.org/
    
    def extract_features4 (self, word, language):
        
        if language == 'english':
            dic = pyphen.Pyphen (lang = 'en')
        
        else: #spanish
            dic = pyphen.Pyphen (lang = 'es')
            
        syllable = dic.inserted(word)
        countSyll = len(syllable.split('-'))
            
        return [countSyll]
            
            
    # Combination of Multiple Features
    def extract_features (self, word, language):
        
        if language == 'english':
            
            # Frequency Count with reference to corpus
            countReuters = self.corpusReuters[word]
            countWordNet = self.corpusWordNet[word]
            
            # Length of words & phrases
            lenWords = len(word)
            lenTokens = len(word.split(' '))
            
            # Syllables Count
            dic = pyphen.Pyphen (lang = 'en')
            syllable = dic.inserted(word)
            countSyll = len(syllable.split('-'))
            
            return [countReuters, countWordNet, countSyll, lenWords, lenTokens]
        
        else: # spanish
            
            # Frequency Count with reference to corpus
#            countWiki = self.wordsWikiEs.get(word)
#            if countWiki == 'Inf' or 'Nan':
#                countWiki = 0
            
            # Length of words & phrases
            lenWords = len(word)
            lenTokens = len(word.split(' '))
            
            # Syllables Count
            dic = pyphen.Pyphen (lang = 'es')
            syllable = dic.inserted(word)
            countSyll = len(syllable.split('-'))
            
            return [countSyll, lenWords, lenTokens]
        
    
    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word'], self.language))
            y.append(sent['gold_label'])

        self.model2.fit(X, y)


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word'], self.language))

        return self.model2.predict(X)
