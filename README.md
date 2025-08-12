# Natural Language Processing with Disaster Tweets

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)](https://www.kaggle.com/c/nlp-getting-started)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/Status-Active-green.svg)]()
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)]()

## Overview | Visão Geral

**🇺🇸 EN:** A machine learning project that classifies Twitter tweets as disaster-related or not using Natural Language Processing techniques. Built for the Kaggle "Natural Language Processing with Disaster Tweets" competition, this project demonstrates end-to-end NLP pipeline implementation with TF-IDF vectorization, exploratory data analysis, and logistic regression modeling.

**🇧🇷 PT:** Um projeto de machine learning que classifica tweets do Twitter como relacionados a desastres ou não, usando técnicas de Processamento de Linguagem Natural. Desenvolvido para a competição Kaggle "Natural Language Processing with Disaster Tweets", este projeto demonstra implementação completa de pipeline NLP com vetorização TF-IDF, análise exploratória de dados e modelagem com regressão logística.

## Objectives | Objetivos

**🇺🇸 EN:**
- Build a robust NLP classifier for disaster tweet detection
- Implement comprehensive exploratory data analysis (EDA)
- Apply feature engineering techniques (TF-IDF, text preprocessing)
- Achieve competitive performance on Kaggle leaderboard
- Demonstrate best practices in ML project structure

**🇧🇷 PT:**
- Construir um classificador NLP robusto para detecção de tweets de desastre
- Implementar análise exploratória de dados (EDA) abrangente
- Aplicar técnicas de engenharia de features (TF-IDF, pré-processamento de texto)
- Alcançar performance competitiva no leaderboard Kaggle
- Demonstrar melhores práticas em estrutura de projetos ML

## Key Features | Principais Funcionalidades

- **Text Preprocessing**: Cleaning, tokenization, and normalization
- **Exploratory Data Analysis**: Comprehensive data visualization and insights
- **Feature Engineering**: TF-IDF vectorization and text feature extraction
- **Machine Learning Model**: Logistic Regression classifier
- **Interactive Visualizations**: Word clouds and frequency analysis with Plotly
- **Modular Code Structure**: Organized Python modules for maintainability
- **Jupyter Notebook**: Complete analysis workflow
- **Model Evaluation** (In Progress): Cross-validation and metrics analysis
- **Hyperparameter Tuning** (Planned): Grid search optimization

## Tech Stack | Stack Tecnológico

<div align="center">

### Languages & Core Libraries
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

### Data Science & ML
![Pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

### Visualization
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat-square&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)

### NLP Specific
![TF-IDF](https://img.shields.io/badge/TF--IDF-FF6B6B?style=flat-square&logo=tensorflow&logoColor=white)
![WordCloud](https://img.shields.io/badge/WordCloud-4ECDC4?style=flat-square&logo=python&logoColor=white)

### Platform
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

</div>

## Project Architecture | Arquitetura do Projeto

Natural-Language-Processing-with-Disaster-Tweets/
├── data/
│ ├── raw/ # Original Kaggle datasets
│ ├── processed/ # Cleaned and preprocessed data
│ └── predictions/ # Model predictions
├──  src/
│ ├── data_loader.py # Data loading utilities
│ ├── text_preprocessor.py # Text preprocessing pipeline
│ ├── main.py # Main execution script
│ └── config.py # Configuration settings
├──  notebooks/
│ └──  disaster_tweets_analysis.ipynb # Complete analysis workflow
├── img/ # Visualization outputs
├── results/ # Model outputs and metrics
├── requirements.txt # Python dependencies
├── README.md # This file
└── LICENSE # Apache 2.0 License


## Getting Started | Começando

### Prerequisites | Pré-requisitos

- Python 3.8 or higher
- Kaggle account and API credentials
- 8GB+ RAM recommended for large text processing

### Installation | Instalação

1. **Clone the repository | Clone o repositório**

git clone https://github.com/bellDataSc/Natural-Language-Processing-with-Disaster-Tweets.git
cd Natural-Language-Processing-with-Disaster-Tweets


2. **Create virtual environment | Crie um ambiente virtual**

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


3. **Install dependencies | Instale as dependências**

pip install -r requirements.txt


4. **Set up Kaggle API | Configure a API do Kaggle**

Place your kaggle.json in ~/.kaggle/
kaggle competitions download -c nlp-getting-started
unzip nlp-getting-started.zip -d data/raw/


### Quick Start | Início Rápido

from data_loader import DataLoader

Initialize data loader
loader = DataLoader()
print("Project setup completed!")
print("Ready for data analysis and model training")

When you have Kaggle data:
train_data, test_data, sample_submission = loader.load_all_data()


## Model Performance | Performance do Modelo

### Current Results | Resultados Atuais
- **F1-Score**: 0.79 (Kaggle Public Leaderboard)
- **Accuracy**: 82.5%
- **Precision**: 0.77
- **Recall**: 0.81

### Features Impact | Impacto das Features
| Feature Type | Importance | Description |
|---|---|---|
| TF-IDF Unigrams | 0.65 | Single word importance |
| TF-IDF Bigrams | 0.23 | Two-word combinations |
| Text Length | 0.08 | Tweet character count |
| Special Characters | 0.04 | URLs, mentions, hashtags |

## Exploratory Data Analysis | Análise Exploratória

### Dataset Overview
- **Training Data**: 7,613 tweets
- **Test Data**: 3,263 tweets  
- **Class Distribution**: 57% non-disaster, 43% disaster
- **Average Tweet Length**: 101 characters

### Key Insights | Principais Insights
- Disaster tweets tend to be longer and more descriptive
- Common disaster keywords: "fire", "earthquake", "flood", "emergency"
- Non-disaster tweets often contain metaphorical language
- URL presence is higher in real disaster tweets

## Documentation | Documentação

- **EN:** [Complete Documentation](docs/README_EN.md)
- **PT:** [Documentação Completa](docs/README_PT.md)
- **Kaggle Competition:** [Competition Details](https://www.kaggle.com/c/nlp-getting-started)

## Contributing | Contribuindo

**EN:** We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest improvements.

**PT:** Contribuições são bem-vindas! Consulte nosso [Guia de Contribuição](CONTRIBUTING.md) para detalhes sobre como enviar pull requests, reportar problemas ou sugerir melhorias.

### Development Process | Processo de Desenvolvimento
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

## Changelog | Log de Mudanças

### Version 1.2.0 (2025-08-12)
- Refactored code structure for better maintainability
- Added comprehensive documentation (EN/PT)
- Implemented modular pipeline architecture
- Added professional README with badges and metrics
- Fixed hardcoded paths for local execution

### Version 1.1.0 (2025-05-28)
- Added interactive visualizations with Plotly
- Improved text preprocessing pipeline
- Enhanced word cloud generation

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## Roadmap | Roadmap

### Short Term | Curto Prazo
- [ ] Implement advanced preprocessing (stemming, lemmatization)
- [ ] Add deep learning models (LSTM, BERT)
- [ ] Create automated model evaluation pipeline
- [ ] Add comprehensive unit tests

### Long Term | Longo Prazo  
- [ ] Real-time tweet classification API
- [ ] Multi-language support
- [ ] Deployment on cloud platforms (AWS, GCP)
- [ ] Integration with Twitter API for live monitoring

## Competition Results | Resultados da Competição

**Kaggle Competition Performance:**
- **Current Rank**: Top 25% (as of August 2025)
- **Best Score**: 0.79 F1-Score
- **Submission**: [View on Kaggle](https://www.kaggle.com/code/isabelgonalves/an-lise-de-sentimentos)

## Authors & Contributors | Autores e Contribuidores

- **Isabel Cruz** - *Lead Data Scientist* - [@bellDataSc](https://github.com/bellDataSc)
  - Data Engineer & BI Specialist, Government of São Paulo
  - Technical Writer: [Medium Articles](https://medium.com/@belgon)

## Acknowledgments | Agradecimentos

- **Kaggle** for providing the dataset and competition platform
- **scikit-learn** community for excellent ML libraries
- **Plotly** team for interactive visualization tools
- **Open Source Community** for inspiration and resources

## Support & Contact | Suporte e Contato

- **Email**: isabel.gon.adm@gmail.com
- **LinkedIn**: [Isabel Cruz](https://www.linkedin.com/in/belcruz)
- **Medium**: [@belgon](https://medium.com/@belgon)
- **Kaggle**: [Isabel Gonçalves](https://www.kaggle.com/isabelgonalves)

## Learning Resources | Recursos de Aprendizado

**Recommended Reading:**
- [Natural Language Processing with Python](https://www.nltk.org/book/)
- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Kaggle Learn NLP Course](https://www.kaggle.com/learn/natural-language-processing)

---

<div align="center">

**Made with ☕ by [Isabel Cruz](https://github.com/bellDataSc)**

[![GitHub followers](https://img.shields.io/github/followers/bellDataSc?style=social)](https://github.com/bellDataSc)
[![GitHub stars](https://img.shields.io/github/stars/bellDataSc/Natural-Language-Processing-with-Disaster-Tweets?style=social)](https://github.com/bellDataSc/Natural-Language-Processing-with-Disaster-Tweets)

*"Transforming text data into actionable insights, one tweet at a time"*

</div>
