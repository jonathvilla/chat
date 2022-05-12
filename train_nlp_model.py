import warnings
import pickle
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
warnings.filterwarnings("ignore")


# " LOAD DATA BASE"
dataset = 'KnowledgeBase.xlsx'
knowledge_base = pd.read_excel(dataset)

# label encode the target variable to transform non-numerical labels
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(knowledge_base["Intent"])  # numerical labels
intent_names = encoder.classes_

" WORD2VECT TRANSFORM AND FEATURE EXTRACTION"
" TF-IDF score represents the relative importance of a term in the document and the entire corpus. "
x = knowledge_base["Utterance"]
tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)
tfidf_vect.fit(x)
x_tfidf = tfidf_vect.transform(x)


" MODEL TRAINING AND DEVELOPMENT"
nfolds = 10
tuned_parameters = {'C': [0.001, 0.10, 0.1, 10, 100, 1000]}
lr_model = GridSearchCV(LogisticRegression(), tuned_parameters, cv=nfolds,  scoring='accuracy') 
lr_model.fit(x_tfidf, y)

means = lr_model.cv_results_['mean_test_score']
stds = lr_model.cv_results_['std_test_score']
parameters = lr_model.cv_results_['params']
results = pd.concat((pd.DataFrame.from_dict(parameters), pd.DataFrame(means, columns=['Mean']), pd.DataFrame(stds, columns=['STD'])), axis=1)

best_c = lr_model.cv_results_['params'][lr_model.best_index_]['C']

# TRAIN MODEL WITH ALL DATA FOR PRODUCTION
best_lr_model = LogisticRegression(C=best_c).fit(x_tfidf, y)

# "SAVE FINAL MODEL"
pickle.dump(best_lr_model, open('nlp_model.pkl', 'wb'))
pickle.dump(tfidf_vect, open('knowledgebase_vocabulary.pkl', 'wb'))
pickle.dump(intent_names, open('intent_names.pkl', 'wb'))
