# LDA-based Topic Modeling Engine for News Articles

This project showcases an advanced topic modeling engine that utilizes Latent Dirichlet Allocation (LDA) to extract key themes from large-scale news datasets.

## Features
- Implements Latent Dirichlet Allocation (LDA) for topic modeling
- Incorporates state-of-the-art summarization techniques, such as TextRank and BART
- Provides concise and coherent summaries of identified topics
- Handles vast amounts of unstructured text data
- Delivers meaningful insights for enhanced user engagement and content discovery

## Technical Skills
- Natural Language Processing (NLP)
- Unsupervised Learning
- Topic Modeling
- Text Summarization
- Web Scraping
- Python
- Libraries: gensim, nltk, requests, BeautifulSoup

## Applications
- News Article Analysis
- Content Recommendation Systems
- Topic Discovery and Exploration
- Text Mining and Information Retrieval

## Methodology
1. **Text Extraction**: The news article text is extracted using web scraping techniques with the `requests` and `BeautifulSoup` libraries.
2. **Text Preprocessing**: The extracted text undergoes preprocessing steps such as removing stopwords, tokenization, and lemmatization using the `nltk` library.
3. **Topic Modeling**: Latent Dirichlet Allocation (LDA) is applied to the preprocessed text using the `gensim` library to discover latent topics within the news article.
4. **Summarization**: Various summarization techniques, including Gensim, Summa, TextRank, LexRank, Luhn, LSA, BART, PEGASUS, and T5, are employed to generate concise summaries of the identified topics.
5. **Evaluation**: The generated summaries are evaluated for coherence, informativeness, and relevance to the original news article.

## Results
- The LDA-based topic modeling engine successfully extracted key themes from the news article, providing insights into the main topics discussed.
- The summarization techniques generated concise and informative summaries of the identified topics, capturing the essential information from the article.
- The BART and PEGASUS models produced the most coherent and fluent summaries compared to other techniques, demonstrating their effectiveness in abstractive summarization.
- The combination of topic modeling and summarization enables efficient content analysis and recommendation systems for large-scale news datasets.

## Getting Started
1. Open the Colab notebook: [LDA-based Topic Modeling Engine](https://colab.research.google.com/drive/1w1taH8fvVQ7sYTsBpnDuGizrGMOqk-5y?usp=sharing)
2. Upload your news dataset in the appropriate format.
3. Follow the instructions in the notebook to run the topic modeling engine and explore the generated topic summaries and insights.

## Contributing
Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. This means you are free to use, modify, and distribute the software, as long as you include the original license and copyright notice in any copies or substantial portions of the software. 

## Contact
For any inquiries or collaborations, please contact:
- Mohamed Oussama NAJI
- LinkedIn: [Mohamed Oussama Naji](https://www.linkedin.com/in/oussamanaji/)
