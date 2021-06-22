Abstractive text-summarization



# Text Summarization #



<!-- ABOUT THE PROJECT -->
## About The Project

Text summarization is considered as a challenging task in the NLP community. The availability of datasets for the task of multilingual text summarization is rare, and such datasets are difficult to construct. This article will show you how to work on the problem of text summarization to create relevant summaries for product reviews about fine food sold on the world’s largest e-commerce platform, Amazon using basic character-level sequence-to-sequence (seq2seq) model by defining the encoder-decoder recurrent neural network (RNN) architecture. 

Abstractive text summarization aims to shorten long text documents into a human readable form that contains the most important facts from the original document



### Built With

Tools and framework used in the project. Here are a few examples.
* [python](https://python.org)
* [keras](https://keras.io/)
* [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [nltk](https://www.nltk.org/)
* [AttentionLayer](https://github.com/thushv89/attention_keras/blob/master/layers/attention.py)
* [sklearn](https://scikit-learn.org/)



<!-- GETTING STARTED -->
## Getting Started

Step by step guide of how project works

### Dataset

Download amazon food review dataset from 'Review.csv' file from [here](https://www.kaggle.com/snap/amazon-fine-food-reviews)


### Prerequisites

Framework
* [tensorflow](https://www.tensorflow.org/)


### Installation

1. Get a dataset from [here](https://www.kaggle.com/snap/amazon-fine-food-reviews) and attention layer from [here](https://github.com/thushv89/attention_keras/blob/master/layers/attention.py)
2. Clone the repo
   ```sh
   git clone https://github.com/parthvalani/Text-summarization.git
   ```
3. Install python packages
   ```sh
   pip install pandas numpy bs4 nltk keras matplotlib sklearn
   ```
4. Run [textsummarization.py](https://github.com/parthvalani/Text-summarization/blob/master/textsummarization.py)

<!-- USAGE EXAMPLES -->
## Methodology

* load and read the data using pandas
* drop null values
* data preprocessing using nltk: renmove stopword, tokenize etc.
* take out the emotion of the tweet using emot
* get tweet sentiment using textblob
* built a model using enocoder-decoder LSTM for training-testing
* visualize it using matplotlib
* predict summary of the review


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact
Parth Valani - [@parth_valani](https://www.linkedin.com/in/parthvalani/) - parthnvalani@gmail.com

Project Link: [https://github.com/parthvalani/Text-summarization](https://github.com/parthvalani/Text-summarization/)
