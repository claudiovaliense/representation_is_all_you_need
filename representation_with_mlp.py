
#imports
import pickle
import numpy
import torch
from torch.utils import data # trabalhar com dados iterados
import random
from transformers import BertModel, BertTokenizer
import pandas as pd
import jsonlines
from transformers import AutoTokenizer
import sklearn
from sklearn import svm
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn # neural network in pytorch
import torch.nn.functional as F
import timeit  # calcular metrica de tempo
import io
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn import preprocessing 
import sys

SEED=42
random.seed(SEED); torch.manual_seed(SEED); numpy.random.seed(seed=SEED) # reproducibily soluction

#hyper-parametros
batch_size=32
lr=0.01
max_epochs=5
limit_patient=5


datasets = [sys.argv[1]]
index_folds = [int(sys.argv[2])]
num_folds = int(sys.argv[3])





# codigo em uma unica celula para executar folds
for dataset in datasets:
    for index_fold in index_folds:
        nome_experimento= f'{dataset}_tfidf_mlp{index_fold}' 
        ini = timeit.default_timer()
        print(nome_experimento)

        # carregar dados
        #ids = pickle.load( open(f'dataset/{dataset}/splits/split_{num_folds}_with_val.pkl', 'rb') )
     
        
        x_train_tfidf, y_train, x_val_tfidf, y_val, x_test_tfidf, y_test = load_svmlight_files([open(f'kaggle_tfidf/{dataset}_tfidf_train{index_fold}', 'rb'),
         open(f'kaggle_tfidf/{dataset}_tfidf_val{index_fold}', 'rb'), open(f'kaggle_tfidf/{dataset}_tfidf_test{index_fold}', 'rb')])

        le = preprocessing.LabelEncoder() # é preciso que tenha label 0 para funcionar abaixo
        le.fit(y_train)
        y_train = le.transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)

        num_labels = len(numpy.unique(y_train))
        print(f'num_labels {num_labels}')

        # carregar dataset no formato do torch
        class CustomDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item =  {'vetor' : torch.tensor(self.encodings[idx,:].todense()).reshape(-1)  }                        
                item['label'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        # tfidf        
        # vectorizer = TfidfVectorizer()
        # x_train_tfidf = vectorizer.fit_transform(list(x_train['text']))
        # x_val_tfidf = vectorizer.transform(list(x_val['text']))
        # x_test_tfidf = vectorizer.transform(list(x_test['text']))
        
        
        train_dataset = CustomDataset(x_train_tfidf, y_train )
        val_dataset = CustomDataset(x_val_tfidf, y_val )
        test_dataset = CustomDataset(x_test_tfidf, y_test )

        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, worker_init_fn=SEED)
        eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, worker_init_fn=SEED)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, worker_init_fn=SEED)
        
        # classe para reducao de dimensionalidade
        class MLP(nn.Module):
          def __init__(self, tamanho_tfidf):
            super().__init__()                
            
            #self.encoder = nn.Sequential(nn.Linear(tamanho_tfidf, tamanho_tfidf), nn.ReLU() ) #encoder 
            #self.softmax = nn.Sequential(   nn.Linear(tamanho_tfidf, num_labels), nn.Softmax(dim=1) )            
            self.softmax = nn.Sequential( nn.Linear(tamanho_tfidf, 768), nn.Linear(768, num_labels), nn.Softmax(dim=1) )            

          def forward(self, X):            
            #encoder = self.encoder( X )
            y_pred = self.softmax(X)
            return y_pred, y_pred #, y_pred

        # definino modelo
        model = MLP(x_train_tfidf.shape[1])                
        loss =  nn.CrossEntropyLoss()          
        optimizer = AdamW(model.parameters(), lr=lr)


        num_training_steps = max_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        # Coloca para o processamento ser feito na GPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        ###
        # Treina o modelo
        progress_bar = tqdm(range(num_training_steps))
        loss_train = []; loss_eval = []; macro_f1_eval = []; micro_f1_eval = []; wei_f1_eval = []
        cont_patient=0; min_loss_eval =10000; max_macro_eval=-1        

        for epoch in range(max_epochs):
            #print(f'Epoch: {epoch}')
            model.train()
            #for batch in train_dataloader: 
            for batch in tqdm(train_dataloader, desc=f'fiting {epoch}'):              
                batch = {k: v.to(device) for k, v in batch.items()}                   
                _,  y_pred = model(batch['vetor'].float())
                l = loss(y_pred, batch["label"].to(torch.int64) )
                l.backward() 
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)       

            loss_train.append(l.item()) #loss treino
            
            # validação
            y_pred_list = []; y_true_list = []
            model.eval() # define para não atualizar pesos na validação
            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}        

                with torch.no_grad(): # define para não atualizar pesos na validação                     
                    _,  y_pred = model(batch['vetor'].float())

                predictions = torch.argmax(y_pred, dim=-1)                   
                l = loss(y_pred,  batch["label"].to(torch.int64) )                                    
                y_pred_list.append(predictions.tolist()) 
                y_true_list.append( list(batch["label"].tolist()) )

            y_pred_batch = []; y_true_batch = []

            for y_batch in y_pred_list: # y_batchs
                for y_doc in y_batch:
                    y_pred_batch.append(y_doc)

            for y_batch in y_true_list: # y_batchs
                for y_doc in y_batch:
                    y_true_batch.append(y_doc)

            #armazena as metricas a partir das predicoes
            macro = sklearn.metrics.f1_score( y_true_batch , y_pred_batch, average='macro')
            micro = sklearn.metrics.f1_score( y_true_batch , y_pred_batch, average='micro')
            wei = sklearn.metrics.f1_score( y_true_batch , y_pred_batch, average='weighted')

            loss_eval_atual = l.item() #f1 loss
            loss_eval.append(loss_eval_atual); macro_f1_eval.append(macro); micro_f1_eval.append(micro); wei_f1_eval.append(wei)

            print(f'Loss: {loss_eval_atual}')
            print(f"Macro-f1: {macro}" )
            print(f"Micro-f1: {micro}" )
            print(f"Weighted-f1: {wei}" )

            # parar de treinar se não houver melhoria  na loss ou macro
            if loss_eval_atual < min_loss_eval or macro > max_macro_eval:
                cont_patient=0
                min_loss_eval = loss_eval_atual
                if macro > max_macro_eval:
                    max_macro_eval = macro
            else:
                cont_patient+=1

            if cont_patient >= limit_patient:
                break
        
        # carregar conjunto todo para transformar os textos
        train_dataset = CustomDataset(x_train_tfidf, y_train )
        val_dataset = CustomDataset(x_val_tfidf, y_val )
        test_dataset = CustomDataset(x_test_tfidf, y_test )
        
        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=batch_size, worker_init_fn=SEED)
        eval_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, worker_init_fn=SEED)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, worker_init_fn=SEED)
        
                
        escreve =  jsonlines.open(f'{nome_experimento}.json', 'w')
        model.eval()
        with torch.no_grad():            
            index_doc=0      
            y_pred_list = [];
            for batch in test_dataloader:        
                batch = {k: v.to(device) for k, v in batch.items()}        
                #x_encod, y_pred0, y_pred1, y_pred = model( batch['vetor'].float() )
                x_encod,  y_pred = model( batch['vetor'].float() )

                # outputs = x_encod.cpu().detach().numpy().tolist()
                # for out in outputs:
                #     doc = {'id' :  ids['test_idxs'][index_fold][index_doc], 'tfidf' : out, 'label' : int(y_test[index_doc])}
                #     #doc = {'id' : index_doc, 'tfidf' : out[0], 'label' : int(X['label'][index_doc])}
                #     escreve.write(doc)
                #     index_doc+=1 

                # predicao do teste por softmax
                predictions = torch.argmax(y_pred, dim=-1) 
                y_pred_list.append(predictions.tolist()) 
                #y_true_list.append( list(batch["label"].tolist()) )

            y_pred_batch = []; y_true_batch = []

            for y_batch in y_pred_list: # y_batchs
                for y_doc in y_batch:
                    y_pred_batch.append(y_doc)

            escreve_pred = jsonlines.open(f'{nome_experimento}_pred.json', 'w')
            doc = {'index_fold' : index_fold, 'y_pred' : y_pred_batch, 'Macro-f1' : sklearn.metrics.f1_score( y_test , y_pred_batch, average='macro'),
                       'Micro-f1' : sklearn.metrics.f1_score( y_test , y_pred_batch, average='micro'),
                       'Weighted-f1' : sklearn.metrics.f1_score( y_test , y_pred_batch, average='weighted'),
                       'time' : timeit.default_timer() - ini
                      }
            escreve_pred.write(doc)

            print("Test Softmax---")
            print(f"Macro-f1: {sklearn.metrics.f1_score( y_test, y_pred_batch, average='macro')}" )
            print(f"Micro-f1: {sklearn.metrics.f1_score( y_test , y_pred_batch, average='micro')}" )
            print(f"Weighted-f1: {sklearn.metrics.f1_score(y_test , y_pred_batch, average='weighted')}" )
            escreve_pred.close()

        escreve.close()


