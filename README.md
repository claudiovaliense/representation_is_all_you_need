### Representation is all you need
Repository article " Representation is all you need "

### Execute MLP tfidf----------------

# Download code
git clone https://github.com/claudiovaliense/representation_is_all_you_need.git

# Create virtual env
virtualenv -p python3 ./venv

# Activate the virtualenv
source ./venv/bin/activate

# Install dependecies
pip install -r requirements.txt

# Download dataset kaggle
kaggle datasets download -d claudiovaliense/datasets-tfidf

Change the command inside the 'representation_with_mlp.sh' file as per the dataset
varios_datasets 'aisopos_ntua_2L' '0' '10'

# Execute soluction
sh representation_with_mlp.sh


### Link to the representations used in the article:

TF-IDF

https://www.kaggle.com/datasets/claudiovaliense/datasets-tfidf

fasttext

https://www.kaggle.com/datasets/claudiovaliense/dataset-fasttext

Zero-Shot BERT

https://www.kaggle.com/datasets/claudiovaliense/datasets-zero-shot-lbd

Fine-Tunned BERT

https://www.kaggle.com/datasets/claudiovaliense/datasets-lbd

https://www.kaggle.com/datasets/claudiovaliense/datasets-bert
