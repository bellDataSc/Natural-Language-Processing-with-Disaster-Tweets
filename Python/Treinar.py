# Dividir os dados de treino em treino e validação
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Treinar o modelo
model = LogisticRegression()
model.fit(X_train_split, y_train_split)

# Prever no conjunto de validação
y_val_pred = model.predict(X_val_split)

# Avaliar o modelo usando F1-score
f1 = f1_score(y_val_split, y_val_pred)
print(f'F1 Score: {f1}')
