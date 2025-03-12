## 🔹 Melhorias Implementadas:¶
## Distribuição das classes (sem seaborn)

## Distribuição do comprimento dos textos

## Nuvem de palavras para cada classe

## Gráfico interativo de palavras mais frequentes


# Importar Bibliotecas Adicionais
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
from collections import Counter

# 1️⃣ Distribuição das Classes (Usando Matplotlib)
class_counts = np.bincount(train_data['target'])

plt.figure(figsize=(6, 4))
plt.barh(['Não relacionado a desastre', 'Relacionado a desastre'], class_counts, color=['blue', 'red'])
plt.xlabel('Quantidade')
plt.title('Distribuição das Classes')
plt.show()

# 2️⃣ Distribuição do Comprimento dos Textos
train_data['text_length'] = train_data['text'].apply(len)

plt.figure(figsize=(8, 5))
plt.hist(train_data['text_length'], bins=30, color='purple', alpha=0.7)
plt.xlabel('Comprimento do Texto')
plt.ylabel('Frequência')
plt.title('Distribuição do Tamanho das Mensagens')
plt.show()

# 3️⃣ Nuvem de Palavras por Classe
def gerar_nuvem_texto(texto, titulo):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(texto))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(titulo)
    plt.show()

# Criar nuvem de palavras para cada classe
gerar_nuvem_texto(train_data[train_data['target'] == 0]['text'], 'Palavras Frequentes - Não Relacionadas a Desastres')
gerar_nuvem_texto(train_data[train_data['target'] == 1]['text'], 'Palavras Frequentes - Relacionadas a Desastres')

# 4️⃣ Gráfico Interativo com Plotly - Palavras Mais Frequentes
def obter_palavras_frequentes(textos, n=20):
    palavras = ' '.join(textos).split()
    contagem = Counter(palavras)
    return contagem.most_common(n)

# Coletar palavras mais frequentes
palavras_comuns = obter_palavras_frequentes(train_data['text'])

# Converter em DataFrame (sem usar pandas)
words, counts = zip(*palavras_comuns)

# Criar gráfico interativo
fig = px.bar(x=words, y=counts, labels={'x': 'Palavras', 'y': 'Frequência'}, title='Palavras Mais Frequentes nos Tweets')
fig.show()
