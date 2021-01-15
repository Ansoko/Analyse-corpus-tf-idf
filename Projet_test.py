# -*- coding: utf-8 -*-
"""
@author: Anne-Sophie Koch, Tom Duffes
"""

import Projet_classes
import pandas as pd

#création de corpus
corpusTest1 = Projet_classes.Corpus("Corpus test 1")
corpusTest2 = Projet_classes.Corpus("Corpus test 2")

doc1=Projet_classes.Document("Les Archives de Sherlock Holmes", "", "Conan Doyle", "When you have eliminated all which is impossible, then whatever remains, however improbable, must be the truth. (common)", '')
doc2=Projet_classes.Document("And Then There Were None", "", "Agatha Christie", "Dr. Armstrong . . . raised the wig. It fell to the floor, revealing the high bald forehead with, in the very middle, a round stained mark from which something had trickled . . . Dr. Armstrong . . . said—and his voice was expressionless, dead, far away: “He’s been shot.” (Word, common)", '')
doc3=Projet_classes.Document("And Then There Were None (2)", "", "Agatha Christie", "Mr. Owen could only come to the island in one way. It is perfectly clear. Mr. Owen is one of us. (word, common)", '')

corpusTest1.add_doc(doc1)
corpusTest2.add_doc(doc2)
corpusTest2.add_doc(doc3)

#test donneesTraitement
#devrait enregistrer une version du texte sans majuscule, sans symbole et avec seulement les racines des mots
print("Traitement des documents")
corpusTest1.donneesTraitement()
print(corpusTest1.collectionAnalyse[0].get_text())
corpusTest2.donneesTraitement()
print(corpusTest2.collectionAnalyse[0].get_text())

#création du vocabulaire
print("Vocabulaire")
corpusTest1.vocabulaire()
print(corpusTest1.voc)
corpusTest2.vocabulaire()
print(corpusTest2.voc)
vocCommun = corpusTest1.voc.union(corpusTest2.voc)
print(vocCommun)

#nombre d'occurences
print("Tableau occurrences")
corpusTest2.occurences()
corpusTest2.freq['Frequence'] = corpusTest2.freq['occurrences'] / corpusTest2.freq['Nbr_docs']
print(corpusTest2.freq.sort_values('occurrences',ascending = False).head(10))

#score tf-idf
print("score tf-idf")
corpusTest1.tfidf(vocCommun)
print(corpusTest1.scoreTfidf)
corpusTest2.tfidf(vocCommun)
print(corpusTest2.scoreTfidf)

#moyenne des scores tf-idf dans chaque corpus
print('moyenne des scores tf-idf')
corpusTest1.moyenne()
print(corpusTest1.moyenneTfidf)
corpusTest2.moyenne()
print(corpusTest2.moyenneTfidf)

#Comparaison
communCorpus = pd.DataFrame(columns=['mot','somme', 'difference']) 
for word in vocCommun:
    if word not in corpusTest1.scoreTfidf.columns :
        somme = float(corpusTest2.moyenneTfidf.loc[corpusTest2.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff=0-float(corpusTest2.moyenneTfidf.loc[corpusTest2.moyenneTfidf['mot']==word,'moyenne_tfidf'])
    elif word not in corpusTest2.scoreTfidf.columns :
        somme = float(corpusTest1.moyenneTfidf.loc[corpusTest1.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff=somme 
    else:
        somme = float(corpusTest1.moyenneTfidf.loc[corpusTest1.moyenneTfidf['mot']==word,'moyenne_tfidf'])+float(corpusTest2.moyenneTfidf.loc[corpusTest2.moyenneTfidf['mot']==word,'moyenne_tfidf'])
        diff = float(corpusTest1.moyenneTfidf.loc[corpusTest1.moyenneTfidf['mot']==word,'moyenne_tfidf'])-float(corpusTest2.moyenneTfidf.loc[corpusTest2.moyenneTfidf['mot']==word,'moyenne_tfidf']) 
    df2 = pd.DataFrame([[word, somme, diff]], columns=['mot', 'somme', 'difference'])
    communCorpus = communCorpus.append(df2, ignore_index=True)
    
communCorpus['plus_important']=0
for index, x in communCorpus.iterrows():
    if x.difference>0:
        communCorpus.loc[[index],'plus_important'] = "Test1"
    elif x.difference<0:
        communCorpus.loc[[index],'plus_important'] = "Test2"
    else:
        communCorpus.loc[[index],'plus_important'] = "Pas important"
communCorpus['difference']=abs(communCorpus['difference'])

#Le mots les plus importants des deux corpus
print(communCorpus.sort_values('somme',ascending = False).head(30))
#les mots communs les plus importants dans un corpus par rapport à l'autre
print(communCorpus.sort_values('difference',ascending = False).head(30))

#vocabulaire exclusif à un corpus
print("Vocabulaire exclusif")
vocTest1 = corpusTest1.voc.difference(corpusTest2.voc)
print(vocTest1)
vocTest2 = corpusTest2.voc.difference(corpusTest1.voc)
print(vocTest2)

#vocabulaire commun aux deux corpus
print("Vocabulaire commun")
vocDeuxCorpus = vocCommun.difference(vocTest1).difference(vocTest2)
print(vocDeuxCorpus)

