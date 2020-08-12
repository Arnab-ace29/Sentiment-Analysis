# Twitter Sentiment Analysis NLP  

### Problem Statement  
Sentiment analysis remains one of the key problems that has seen extensive application of natural language processing. It helps in detecting the sentiment o people on internet regarding a product, company or any scenario.
### Data Description  
For training the models, a labelled tweets dataset is provided. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet. The test data file contains only tweet ids and the tweet text with each tweet in a new line.

### Approach and Implementation  
Three methods have been used to analyse the dataset.  
1. Using **BERT** embeddings  
2. Using **Bi-LSTM** 
3. Using **Bi-LSTM** and **Bi-GRU** and concatinating their results
For 2. and 3. the preprocessing steps are same. First, we remove all the links(eg. https, www etc) and clean the data. We also remove all the puncuations, emoticons, replace many commonly used internet shortcuts like (u-you, ttyl-talk to you later etc), and perform speelcheck using a library called spell check(Spell Check takes 2+hrs and it may be avoided).  
We create another dataframe which contains the **features** data, which is derived from the main dataset.
     - Length of the Tweet
     - Average length of words in a tweet 
     - Number of characters in a tweet
     - Number of Stopwords in a tweet 
     - Number of puncuations in a tweet 
     - Number of mentions(@) in a tweet
     - Number of links in a tweet(tweet containing http,https, www)
     - Number of # in a tweet
The tweets are embedded using **Glove** Embeddings. Here I have used tweet.300Dwhich is trained on twitter dataset. The tweet data is embedded and is passed through layers of Neural networks,and in the penultimate layer we concatinate the features dataset, nd finally pass through sigmoid layers. The activation functions used here is **"SELU"** and initilization is **"HE-initilization"**.   
For 2-3 the process till here is same, but in 3 we similarly train model on **bi-GRU**, and the result is concatinated.  

For BERT Embeddings, we train similarly but we will skip the features dataset process, and Pass the pre-processed tweets directly to the **BERT** embeddings layers.  

### Results  
The results obtained from BERT is the best amongst all three models. There is a slight difference between 2 and 3 but 3 performs better. Currently this model score stands on **rank 98 out of 6000+ participants with 865 unique solutions**.  

### Further Scope of Study
- The results may further be refined using other combinations of word embeddings such as GloVe, Word2Vec and fastText
- Other Glove embeddings can be used such as Glove wikipedia, Glove.840b etc
- Ensemble of different learning algorythms might be used to increase the prediction accuracy of models further

