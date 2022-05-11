import jsonlines
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestCentroid
import pickle
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
from sklearn import preprocessing



macro_list = []; micro_list = []
classificador='gbm' # 'svm' 'centroide'

#datasets = ['acm', 'reut']
datasets = ['20ng']
datasets = ['aisopos_ntua_2L', 'debate_2L', 'english_dailabor_2L', 'nikolaos_ted_2L', 'pang_movie_2L', 'sanders_2L', 'sarcasm_2L', 'sentistrength_bbc_2L', 'sentistrength_digg_2L', '20ng', 'acm', 'reut', 'webkb']
datasets = ['aisopos_ntua_2L', 'debate_2L', 'english_dailabor_2L', '20ng', 'acm', 'reut', 'webkb']
datasets = ['vader_movie_2L', 'yelp_review_2L', 'books', 'dblp', 'vader_nyt_2L']
#datasets = ['aisopos_ntua_2L']
# dd = ['20ng',  'acm',  'aisopos_ntua_2L', 'books', 'dblp',  'pang_movie_2L', 'reut', 'sst2', 'webkb', 'vader_movie_2L', 'vader_nyt_2L', 'yelp_review_2L']
datasets = ['20ng', 'acm', 'webkb', 'reut', 'aisopos_ntua_2L']

#for classificador in ['centroide', 'lr', 'knn', 'gbm', 'rf', 'svm']:
for classificador in ['centroide', 'rf', 'svm']:
  for dataset in datasets:
      for index_fold in range(0,10):
          nome_experimento= f'{dataset}_bert{index_fold}'
          print(nome_experimento+classificador)

          #for fold_temp in  range(3,10):
          #    nome_experimento= f'reut_bert{fold_temp}'
          #    for line in jsonlines.open(f'{nome_experimento}_pred.json'):
          #        print(str(line['time']).replace('.','.'))
          #        #print(line['Macro-f1'])

          ids = pickle.load( open(f'dataset/{dataset}/splits/split_10_with_val.pkl', 'rb') )
          #ids = pickle.load( open(f'dataset/webkb/splits/split_10_with_val.pkl', 'rb') )

          webkb = jsonlines.open(f'{nome_experimento}.json')
          #dataset='../input/webkb-bert/webkb'
          #index_fold=1
          #webkb = jsonlines.open(f'{dataset}_bert{index_fold}.json') # testar a representacao do bert original
          #webkb = jsonlines.open(f'{dataset}_{index_fold}.json') # testar a representacao do bert original

          ids_train = list(ids['train_idxs'][index_fold])

          docs = []
          for line in webkb:
            docs.append( (line['id'], line['bert'], line['label']) )
            
          X = pd.DataFrame(docs, columns=['id', 'bert', 'label'])

          #le = preprocessing.LabelEncoder() # é preciso que tenha label 0 para funcionar abaixo
          #le.fit(labels)
          #labels = le.transform(labels)

          #x_train = X.iloc[  ids['train_idxs'][index_fold] ] # seleciona a partir de indices
          #x_val = X.iloc[  ids['val_idxs'][index_fold] ]
          #x_test = X.iloc[  ids['test_idxs'][index_fold] ]        

          x_train = X.query(f"id == {ids['train_idxs'][index_fold]}")
          x_val = X.query(f"id == {ids['val_idxs'][index_fold]}")
          x_test = X.query(f"id == {ids['test_idxs'][index_fold]}")

          if classificador == 'centroide':
              estimator = NearestCentroid()
          if classificador == 'gbm':
            estimator = lgb.LGBMClassifier()
          if classificador == 'svm':
            estimator = svm.LinearSVC(random_state=42, max_iter=1000)
          if classificador == 'rf':
            estimator = RandomForestClassifier (n_estimators=200, n_jobs=-1)          
          if classificador == 'lr':
            estimator =LogisticRegression(random_state=42, solver='liblinear',n_jobs=-1)
          if classificador == 'knn':
            estimator = KNeighborsClassifier(n_neighbors=3,  n_jobs=-1)

          #estimator = NearestCentroid()
          #estimator = RandomForestClassifier (n_estimators=200)
          #estimator = svm.LinearSVC(random_state=42, max_iter=1000)
          #estimator =LogisticRegression(random_state=42, solver='liblinear')
          #estimator = KNeighborsClassifier(n_neighbors=3)
          #estimator = svm.LinearSVC(random_state=42, max_iter=1000, C=0.001)
          #estimator = lgb.LGBMClassifier()
          #estimator = GridSearchCV(estimator, [{'n_estimators':  [10, 50, 100, 200]}], cv=5, scoring='f1_macro', n_jobs=-1)
          #estimator = GridSearchCV(estimator, [{'C':  [0.001, 0.01, 0.1, 1, 10, 100] }], cv=5, scoring='f1_macro', n_jobs=-1)
          #estimator = GridSearchCV(estimator, [{'C':  [0.01, 0.1, 1] }], cv=5, scoring='f1_macro', n_jobs=-1)
          #estimator = GridSearchCV(estimator, [{'n_neighbors':  [3, 5, 10] }], cv=3, scoring='f1_macro', n_jobs=-1)
          



          estimator.fit(list(x_train['bert']), list(x_train['label']))

          #print(f'estimator.best_params_ :\n {estimator.best_params_}')

          y_pred = estimator.predict(list(x_test['bert']))
          y_pred = y_pred.tolist()
          macro_list.append( sklearn.metrics.f1_score(list(x_test['label']), y_pred, average='macro') )
          micro_list.append( sklearn.metrics.f1_score(list(x_test['label']), y_pred, average='micro') )
          #print(f"Micro-f1: { sklearn.metrics.f1_score(list(x_test['label']), y_pred, average='micro')}" )
          escreve = jsonlines.open(f'{nome_experimento}_{classificador}', 'w')
          doc = {'index_fold' : index_fold, 'y_pred' : y_pred, 'Macro-f1' : sklearn.metrics.f1_score( list(x_test['label']) , y_pred, average='macro'),
            'Micro-f1' : sklearn.metrics.f1_score( list(x_test['label']) , y_pred, average='micro'), 'Weighted-f1' : sklearn.metrics.f1_score( list(x_test['label']) , y_pred, average='weighted') }
          escreve.write(doc)

