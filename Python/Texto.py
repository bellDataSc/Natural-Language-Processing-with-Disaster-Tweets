#Pré-processamento de Texto
# Preencher valores faltantes

  train_data['text'] = train_data['text'].fillna('')
  test_data['text'] = test_data['text'].fillna('')

# Transformar o texto em vetores numéricos usando TF-IDF

  vectorizer = TfidfVectorizer(max_features=5000)
  X_train = vectorizer.fit_transform(train_data['text'])
  X_test = vectorizer.transform(test_data['text'])

# Separar os rótulos

  y_train = train_data['target']
