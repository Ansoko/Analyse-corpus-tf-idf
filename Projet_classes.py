#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Anne-Sophie Koch, Tom Duffes
"""

################################## Déclaration des classes ##################################

import pickle

import re
import pandas as pd

import math

# import nltk
# print('stopwords et punkt à télécharger')
# nltk.download()

class Corpus():
    
    def __init__(self,name):
        self.name = name
        self.collection = {}
        self.authors = {}
        self.id2doc = {}
        self.id2aut = {}
        self.ndoc = 0
        self.naut = 0
        
        #analyse
        self.collectionAnalyse = {}
        self.voc = set({})
        self.textAll = ""
        self.freq = pd.DataFrame(columns=['mots', 'occurrences', 'Nbr_docs'])
        self.scoreTfidf = pd.DataFrame(columns=['num_doc'])   
        self.moyenneTfidf = pd.DataFrame(columns=['mot', 'moyenne_tfidf'])
        
    def add_doc(self, doc):
        
        self.collection[self.ndoc] = doc
        self.id2doc[self.ndoc] = doc.get_title()
        self.ndoc += 1
        aut_name = doc.get_author()
        aut = self.get_aut2id(aut_name)
        if aut is not None:
            self.authors[aut].add(doc)
        else:
            self.add_aut(aut_name,doc)
            
    def add_aut(self, aut_name,doc):
        
        aut_temp = Author(aut_name)
        aut_temp.add(doc)
        
        self.authors[self.naut] = aut_temp
        self.id2aut[self.naut] = aut_name
        
        self.naut += 1

    def get_aut2id(self, author_name):
        aut2id = {v: k for k, v in self.id2aut.items()}
        heidi = aut2id.get(author_name)
        return heidi

    def get_doc(self, i):
        return self.collection[i]
    
    def get_coll(self):
        return self.collection

    def __str__(self):
        return "Corpus: " + self.name + ", Number of docs: "+ str(self.ndoc)+ ", Number of authors: "+ str(self.naut)
    
    def __repr__(self):
        return self.name

    def sort_title(self,nreturn=None):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_title())][:(nreturn)]

    def sort_date(self,nreturn):
        if nreturn is None:
            nreturn = self.ndoc
        return [self.collection[k] for k, v in sorted(self.collection.items(), key=lambda item: item[1].get_date(), reverse=True)][:(nreturn)]
    
    def save(self,file):
            pickle.dump(self, open(file, "wb" ))

#TD4                  
    def textAllFunc(self):
        #on colle tous les documents du corpus en une seule variable
        if(self.textAll == ""):
            for i in self.collection:
                self.textAll = self.textAll + self.collection[i].get_text()

    def stats(self, n):
        from collections import Counter
        text = self.textAll
        text = Corpus.nettoyer_texte(text)
        mots = re.findall(r'\w+', text)
        L = Counter(mots)      
        df = pd.DataFrame({'occurrence':L})
        #print("Il y a ",df.shape[0]," mots différents.")
        print(df.sort_values('occurrence', ascending = False).head(n))    
       
#projet
    #nettoyer_texte prend en entrée une chaine de caractère
    #le texte retourné est prêt à être analyser
    def nettoyer_texte(text):
        text = text.lower() #suppression des majuscules
        text = text.replace("\n"," ") #suppression des sauts de ligne
        text = re.sub(r'[^\w\s]',' ',text) #suppression des symboles
        text = re.sub(r'[\d]',' ',text) #suppression des numéros
        
        #Certains mots ont la même racine mais sont conjugés, donc on va les considérer comme étant les mêmes mots
        from nltk.stem import PorterStemmer
        porter = PorterStemmer()
        
        from nltk.corpus import stopwords
        text2 = text.split(" ")
        text=""
        for word in text2:
            if word in stopwords.words('english'):
                #si le mot est un mot générique (the, a, etc...) on l'enlève
                continue
            word = porter.stem(word) #racine du mot
            text += " " + word
        
        text = re.sub(r"\s+", ' ', text) #suppression des doubles espaces
        text = re.sub(r"^\s+|\s+$", '', text) #suppression des espaces de début ou/et de fin      
        return text
    
    #remplit l'ensemble "voc", le vocabulaire du document modifié
    def vocabulaire(self):
        for i in self.collectionAnalyse:
            text = self.collectionAnalyse[i].get_text()
            text = text.split(" ")
            self.voc.update(set(text))
        #print(self.voc)

    
    def donneesTraitement(self):
        import copy
        self.collectionAnalyse = copy.deepcopy(self.collection)
        for i in self.collectionAnalyse:
            self.collectionAnalyse[i].text = Corpus.nettoyer_texte(self.get_doc(i).get_text())
     
    #remplit le tableau freq contenant le nombre d'occurence et le nombre de document où apparait le mot
    def occurences(self):        
        for i in self.collection:
            #tous les mots différents du document
            vocText = set()
            #nombre de mots dans le document
            nbrmotsdoc = 0
            text = self.collectionAnalyse[i].get_text().split(" ")
            for word in text:
                nbrmotsdoc+=1
                if len(self.freq.loc[self.freq['mots']==word,].index) > 0 :
                #si on a déjà rencontré ce mot dans le corpus
                    self.freq.loc[self.freq['mots']==word,'occurrences'] += 1
                    
                    #si c'est la première fois qu'on rencontre ce mot dans le document
                    if not word in vocText:
                        #on ajoute 1 au compteur de document contenant ce mot
                        self.freq.loc[self.freq['mots']==word,'Nbr_docs'] += 1
                        vocText.add(word)
                else:
                    df2 = pd.DataFrame([[word, 1, 1]], columns=['mots', 'occurrences', 'Nbr_docs'])
                    self.freq = self.freq.append(df2, ignore_index=True)
                    vocText.add(word)


    
    #calcul du score tf-idf
    def tfidf(self, voca):
        idf = pd.DataFrame(columns=['mots', 'Nbr_docs']) 
        for i in self.collectionAnalyse:  
            #nouvelle ligne dans le dataframe
            self.scoreTfidf = self.scoreTfidf.append({'num_doc' : i}, ignore_index=True)
            self.scoreTfidf = self.scoreTfidf.fillna(0)    
            
            #vocText regroupe tous les mots différents du document i
            vocText = set()
            
            text = self.collectionAnalyse[i].get_text().split(" ")
            for word in text:
                try:
                #si on a déjà rencontré ce mot dans le corpus
                    #tf
                    self.scoreTfidf.loc[self.scoreTfidf['num_doc']==i,word] += 1
                    #idf
                    if not word in vocText:
                        idf.loc[idf['mots']==word,'Nbr_docs'] += 1
                        vocText.add(word)
                    
                except:
                #sinon on l'ajoute au dataframe
                    #tf
                    self.scoreTfidf.loc[self.scoreTfidf['num_doc']==i,word] = 0
                    self.scoreTfidf = self.scoreTfidf.fillna(0)
                    self.scoreTfidf.loc[self.scoreTfidf['num_doc']==i,word] += 1
                    #idf
                    df2 = pd.DataFrame([[word, 1]], columns=['mots', 'Nbr_docs'])
                    idf = idf.append(df2, ignore_index=True)
                    vocText.add(word)

            self.scoreTfidf.loc[self.scoreTfidf['num_doc']==i,self.scoreTfidf.columns != 'num_doc'] = self.scoreTfidf.loc[self.scoreTfidf['num_doc']==i,self.scoreTfidf.columns != 'num_doc'].apply(lambda x: x/len(text), axis=0)
        print("tf terminé")
        
        #test pour une valeur (armstrong)
        # try:
        #     print(self.scoreTfidf['armstrong'])
        # except:
        #     print()
        
        idf['score_idf']=idf['Nbr_docs'].apply(lambda x: math.log(len(self.collectionAnalyse)/x))
        print("idf terminé")
        
        # try:
        #     print(idf.loc[idf['mots']=='armstrong'])
        # except:
        #     print()
        
        for word in self.scoreTfidf.drop(['num_doc'], axis=1): #pour chaque mot   
            #score tf-idf = score tf * score idf
            self.scoreTfidf[word] = self.scoreTfidf[word].apply(lambda x: float(x)*float(idf.loc[idf['mots']==word,'score_idf']))
        print("tf-idf terminé")
        
        
    #moyenne des scores tfidf des mots du corpus
    def moyenne(self):
        for word in self.scoreTfidf.drop(['num_doc'], axis=1):
            df2 = pd.DataFrame([[word, self.scoreTfidf[word].mean()]], columns=['mot', 'moyenne_tfidf'])
            self.moyenneTfidf = self.moyenneTfidf.append(df2, ignore_index=True)
            
            
    
class Author():
    def __init__(self,name):
        self.name = name
        self.production = {}
        self.ndoc = 0
        
    def add(self, doc):     
        self.production[self.ndoc] = doc
        self.ndoc += 1

    def __str__(self):
        return "Auteur: " + self.name + ", Number of docs: "+ str(self.ndoc)
    def __repr__(self):
        return self.name
    


class Document():
    
    # constructor
    def __init__(self, date, title, author, text, url):
        self.date = date
        self.title = title
        self.author = author
        self.text = text
        self.url = url
    
    # getters
    def get_author(self):
        return self.author

    def get_title(self):
        return self.title
    
    def get_date(self):
        return self.date
    
    def get_source(self):
        return self.source
        
    def get_text(self):
        return self.text

    def __str__(self):
        return "Document " + str(self.getType()) + " : " + self.title
    
    def __repr__(self):
        return self.title
    
    def getType(self):
        pass
    