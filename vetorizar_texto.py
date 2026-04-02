from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

texts = [
    "Esse filme é excelente, gostei muito!",
    "Filme horrível, odiei completamente!",
    "O filme foi bom, mas poderia ser melhor.",
    "Excelente atuação, mas o roteiro é fraco."
]

# Agora você instancia diretamente a classe importada
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

df_tfidf = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(df_tfidf)