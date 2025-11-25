from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo

def vetorizacao(dados):
    tfidf = TfidfVectorizer()
    
    df = dataset_completo()
    x = df["sintomas"].astype(str)

    tfidf.fit(x)

    x_tfidf = tfidf.transform(x)

    return tfidf