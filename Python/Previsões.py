# Prever no conjunto de teste
test_predictions = model.predict(X_test)

# Criar o arquivo de submissão
submission = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission.to_csv('submission.csv', index=False)
