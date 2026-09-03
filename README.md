# AI-Driven Stock Market Sentiment Analysis

This project uses natural language processing and machine learning to analyze the sentiment of stock-market news and explore its relationship with daily stock data. It is designed to help analysts summarize market information and use news sentiment as an additional signal when studying stock-price movements.

## Project Overview

The notebook performs an end-to-end exploratory and modeling workflow:

- Loads and validates historical stock-news data
- Cleans and explores news, price, volume, and sentiment fields
- Analyzes news length, sentiment distribution, and market trends
- Creates Word2Vec document embeddings
- Creates sentence embeddings with `BAAI/bge-base-en-v1.5`
- Trains Random Forest and feed-forward neural-network classifiers
- Evaluates models using accuracy, precision, recall, F1 score, and confusion matrices

The sentiment target is represented by three classes:

- `1`: Positive
- `0`: Neutral
- `-1`: Negative

## Repository Structure

```text
.
├── Project_I_GenAI_Stock.ipynb   # Analysis, visualization, and modeling notebook
├── stock_news_.csv               # Historical news and stock-market dataset
└── README.md                     # Project documentation
```

## Dataset

The included dataset contains 349 records covering January 2, 2019 through April 30, 2019. Each record includes:

| Column | Description |
| --- | --- |
| `Date` | Date on which the news was released |
| `News` | Stock-related news article or text |
| `Open` | Opening stock price |
| `High` | Highest stock price during the day |
| `Low` | Lowest stock price during the day |
| `Close` | Closing stock price |
| `Volume` | Number of shares traded |
| `Label` | News sentiment: positive, neutral, or negative |

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Gensim Word2Vec
- Sentence Transformers
- Hugging Face Transformers and PyTorch
- Scikit-learn
- TensorFlow and Keras

## Getting Started

### Option 1: Google Colab

1. Open `Project_I_GenAI_Stock.ipynb` in Google Colab.
2. Upload `stock_news_.csv` to your Google Drive.
3. Update the dataset path in the notebook to match the location of your file.
4. Run the cells from top to bottom.

The notebook installs its main dependencies in an early cell. The first run may also download the `BAAI/bge-base-en-v1.5` model from Hugging Face.

### Option 2: Local Jupyter Environment

Install the main dependencies:

```bash
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.6.1 scipy==1.13.1 gensim==4.3.3 sentence-transformers==3.4.1 matplotlib seaborn torch tensorflow transformers jupyter
```

Then open the notebook:

```bash
jupyter notebook Project_I_GenAI_Stock.ipynb
```

When running locally, replace the Google Drive loading code with:

```python
News = pd.read_csv("stock_news_.csv")
```

## Modeling Workflow

The notebook compares two text-representation approaches:

1. **Word2Vec embeddings:** averages word vectors to create one feature vector per article.
2. **Sentence-transformer embeddings:** encodes each article with `BAAI/bge-base-en-v1.5`.

These representations are used with:

- Random Forest classification
- A dense neural network with dropout layers

The data is split into training and test sets using an 80/20 split with `random_state=42` for reproducibility.

## Important Notes

- This project is for educational and analytical purposes and is not financial advice.
- News sentiment alone cannot reliably predict future stock prices.
- Results may vary depending on library versions, hardware, random seeds, and downloaded model versions.
- The notebook currently expects a Google Drive path by default; update it when using the CSV included in this repository.

## Future Improvements

- Add time-based validation instead of only a random train-test split.
- Compare predictions with actual future price returns rather than sentiment labels alone.
- Add weekly news aggregation and automatic summarization.
- Tune model hyperparameters and address class imbalance if needed.
- Save trained models and embedding outputs for repeatable inference.
