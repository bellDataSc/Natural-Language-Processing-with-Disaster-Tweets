# Verificar a distribuição das classes
sns.countplot(train_data['target'])
plt.title('Distribuição das Classes')
plt.show()

# Verificar valores faltantes
print(train_data.isnull().sum())
