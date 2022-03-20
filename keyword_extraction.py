import warnings
warnings.filterwarnings("ignore")
with open('text.txt', 'r', encoding="utf-8") as input:
    data = input.readlines()
    input.close()
with open('stopwords.txt', 'r',encoding="utf-8") as sw:
    my_stopwords=[sw.read().replace('\n', ',')]
    sw.close()
#----------------LSA
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words = my_stopwords, encoding="utf-8", lowercase= True,max_features= 1000,  max_df = 0.5, smooth_idf=True)

# Transforming the tokens into the matrix form through .fit_transform()
matrix= vectorizer.fit_transform(data)
# SVD represent documents and terms in vectors
from sklearn.decomposition import TruncatedSVD
SVD_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=100, random_state=42)
SVD_model.fit(matrix)
# Getting the terms 
terms = vectorizer.get_feature_names()
# Iterating through each topic
with open("output.txt", 'a', encoding="utf-8") as output:
    output.write("USING LSA METHOD\n")
    for i, comp in enumerate(SVD_model.components_):
        terms_comp = zip(terms, comp)
        # sorting the 7 most important terms
        sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:10]
        output.write("Topic "+str(i)+": \n")
        for t in sorted_terms:
            output.write(t[0]+" ")
        output.write(' \n')
print ("GHI FILE THANH CONG")      
output.close()

