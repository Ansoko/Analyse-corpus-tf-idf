# -*- coding: utf-8 -*-
"""
@author: Anne-Sophie Koch, Tom Duffes
"""

import Projet_classes

import pandas as pd

################### Interface ###########################
from tkinter import *
import tkinter

#on ouvre une première fenêtre ou on va demander à l'utilisateur les paramètres
fenetre = Tk()

#voici la fonction qui s'occupe de récupérer les paramètres et qui ouvre une box de confirmation
def recupere():
    global motCorpus
    global nbrDocument
    motCorpus=entree.get()
    nbrDocument=entree2.get()
    tkinter.messagebox.showinfo(title='Info', message='Nous allons chercher un subreddit nommé r/'+entree.get()+" ainsi que la liste des documents Arxiv portant sur ce même thème. On va analyser le texte sur les "+str(entree2.get()+' documents les plus regardés.' ))
    fenetre.destroy()
 

#on remplit la fenêtre de widget , avec une Entry et une spinbox ou l'on s'occupe de récupérer les valeurs que l'utilisateurs a donné.                      
label = Label(fenetre, text="Comparaison de Corpus Reddit/Arxiv",font=("Calibri", 18))
label.pack()
label = Label(fenetre, text="     ",font=("Impact", 8))
label.pack()
label = Label(fenetre, text="Entrez un nom de corpus",font=("Arial", 12))
label.pack()
value1 = StringVar() 
value1.set('nutrition,politics,...')
entree = Entry(fenetre, textvariable=value1, width=30)
entree.pack()
label = Label(fenetre, text=" ",font=("Impact", 5))
label.pack()
label = Label(fenetre, text="Choisissez le nombre de documents",font=("Arial", 12))
label.pack()
entree2 = Spinbox(fenetre, from_=0, to=1000)
entree2.pack()

#On utilise un bouton pour appliquer la fonction
bouton = Button(fenetre, text='Valider', command=recupere)
bouton.pack()

#fin de la première fenêtre
fenetre.mainloop()

################### Création du Corpus ##################
import datetime as dt
import praw
import urllib.request
import xmltodict

corpusReddit = Projet_classes.Corpus(motCorpus+" Reddit")
corpusArxiv = Projet_classes.Corpus(motCorpus+" Arxiv")

#Documents reddit
reddit = praw.Reddit(client_id='IqaqZZKaYkjdPA', 
                     client_secret='olWSho7M10zcJz5CQbIFxfs_AtQ', 
                     user_agent='Reddit WebScraping')
hot_posts = reddit.subreddit(motCorpus).hot(limit=int(nbrDocument))
for post in hot_posts:
    datet = dt.datetime.fromtimestamp(post.created)
    txt = post.title + ". "+ post.selftext
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    doc = Projet_classes.Document(datet,
                   post.title,
                   post.author_fullname,
                   txt,
                   post.url)
    corpusReddit.add_doc(doc)

#Documents Arxiv
url = 'http://export.arxiv.org/api/query?search_query=all:'+motCorpus+'&start=0&max_results='+str(nbrDocument)
data =  urllib.request.urlopen(url).read().decode()
docs = xmltodict.parse(data)['feed']['entry']

for i in docs:
    datet = dt.datetime.strptime(i['published'], '%Y-%m-%dT%H:%M:%SZ')
    try:
        author = [aut['name'] for aut in i['author']][0]
    except:
        author = i['author']['name']
    txt = i['title']+ ". " + i['summary']
    txt = txt.replace('\n', ' ')
    txt = txt.replace('\r', ' ')
    doc = Projet_classes.Document(datet,
                   i['title'],
                   author,
                   txt,
                   i['id']
                   )
    corpusArxiv.add_doc(doc)

print("Création du corpus Reddit, %d documents et %d auteurs" % (corpusReddit.ndoc,corpusReddit.naut))
print("Création du corpus Arxiv, %d documents et %d auteurs" % (corpusArxiv.ndoc,corpusArxiv.naut))


################## Analyse #####################

#nettoyage des textes du corpus pour l'analyse
corpusReddit.donneesTraitement()
corpusArxiv.donneesTraitement()

#création du vocabulaire
corpusReddit.vocabulaire()
corpusArxiv.vocabulaire()
vocCommun = corpusReddit.voc.union(corpusArxiv.voc)

#nombre d'occurences
corpusReddit.occurences()
corpusArxiv.occurences()

#score tf-idf
corpusReddit.tfidf(vocCommun)
corpusArxiv.tfidf(vocCommun)

#moyenne des scores tf-idf dans chaque corpus
corpusReddit.moyenne()
corpusArxiv.moyenne()

#Comparaison
communCorpus = pd.DataFrame(columns=['mot','somme', 'difference']) 

#pour chaque mot, on calcule la somme et la différence entre les scores tfidf
for word in vocCommun:
    if word not in corpusReddit.scoreTfidf.columns :
        somme = float(corpusArxiv.moyenneTfidf.loc[corpusArxiv.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff=0-float(corpusArxiv.moyenneTfidf.loc[corpusArxiv.moyenneTfidf['mot']==word,'moyenne_tfidf'])
    elif word not in corpusArxiv.scoreTfidf.columns :
        somme = float(corpusReddit.moyenneTfidf.loc[corpusReddit.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff=somme 
    else:
        somme = float(corpusReddit.moyenneTfidf.loc[corpusReddit.moyenneTfidf['mot']==word,'moyenne_tfidf'])+float(corpusArxiv.moyenneTfidf.loc[corpusArxiv.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff = float(corpusReddit.moyenneTfidf.loc[corpusReddit.moyenneTfidf['mot']==word,'moyenne_tfidf'])-float(corpusArxiv.moyenneTfidf.loc[corpusArxiv.moyenneTfidf['mot']==word,'moyenne_tfidf']) 
    df2 = pd.DataFrame([[word, somme, diff]], columns=['mot', 'somme', 'difference'])
    communCorpus = communCorpus.append(df2, ignore_index=True)

communCorpus['plus_important']=0
for index, x in communCorpus.iterrows():
    if x.difference>0:
        communCorpus.loc[[index],'plus_important'] = "Reddit"
    elif x.difference<0:
        communCorpus.loc[[index],'plus_important'] = "Arxiv"
    else:
        communCorpus.loc[[index],'plus_important'] = "Pas important"        
communCorpus['difference']=abs(communCorpus['difference'])

#vocabulaire exclusif à un corpus
vocReddit = corpusReddit.voc.difference(corpusArxiv.voc)
vocArxiv = corpusArxiv.voc.difference(corpusReddit.voc)

#vocabulaire commun aux deux corpus
df3 = vocCommun.difference(vocReddit).difference(vocArxiv)
#J et H sont utiles pour comparer les deux corpus
J=len(vocCommun)
H=len(df3)

#le premier df va seulement s'occuper de la différence entre les deux, soit un grand score très présent chez l'un, et très faible chez l'autre
df = communCorpus
df=df.sort_values(by='difference',ascending=False)
#df4, aussi afficher,va retrouver le mot avec le plus grand score dans le total des deux corpus
df4=df.sort_values(by='somme',ascending=False)
del df['somme']
del df4['difference']
#df1 au final n'est pas utilisé
df1 = corpusReddit.moyenneTfidf
df1=df1.sort_values(by='moyenne_tfidf',ascending=False)

#☺df2 au final n'est pas utilisé
df2 = corpusArxiv.moyenneTfidf
df2=df2.sort_values(by='moyenne_tfidf',ascending=False)

#on ouvre la fenêtre d'affichage des resultats
root = Tk() 
#on voulait afficher les datas frame via tkinter, et la seule solution, trouvée sur internet, est celle-ci 
t1 = Text(root) 
t1.pack() 

class PrintToT1(object): 
    def write(self, s): 
        t1.insert(END, s) 

sys.stdout = PrintToT1() 

#le problème avec ce code, c'est qu'il est impossible de manipuler le data frame d'ou l'affichage très limité
label = Label(root, text="Comparaison de Corpus Reddit/Arxiv",font=("Calibri", 14))
label.pack()
print('Tableau Comparatif \n')
print('Voici les 30 premiers mots ayant une différence de score TF-IDF élevé entre les deux corpus.')
print (df.head(30))

print('Tableau du mot le plus présent \n')
print("Voici les mots les plus importants dans l'ensemble des deux corpus(la somme des scores tf-idf), ainsi que les mots avec le plus petit score combiné.")
print (df4)

print("\n\nLe nombre de mot en commun dans les deux corpus est de " + str(H)+" sur les "+str(J)+" mots uniques que comportent les deux corpus.\n")
print("Voici la liste des mots en commun : \n")
print(df3)

print("\n\nVoici la liste des 30 premiers scores de ces mots :\n")
print(communCorpus[communCorpus['mot'].isin(list(df3))].sort_values(by='somme',ascending=False).head(30))

#print('Tableau Reddit')
#print('Voici les 30 mots avec le plus grand TF-IDF score, ainsi que les 30 mots avec le plus faible score.')
#print (df1)
#print('Tableau Arxiv')
#print('Voici les 30 mots avec le plus grand TF-IDF score, ainsi que les 20 mots avec le plus faible score.')
#print(df2)
#mainloop()
root.mainloop()
root.destroy()

