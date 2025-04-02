## Model Performance
Each of the 3 selected classification algorithms **initially** performed nearly identical to one another. SVM, K-Nearest Neighbors, and Logistic Regression all had F1 scores around .965, showing strong performance across the board.

After tweaking some hyperparameters, I was actually able to get a perfect F1 score of 1.0 using KNN with n_neighbors=6. So while all models were solid choices for this clean and simple dataset, K-Nearest Neighbors ended up performing the best after tuning.

Support Vector Machines are still arguably a great choice due to their flexibility with non-linear data, but in this case, KNN edges ahead when optimized.

## Performance Considerations
There’s a lot that can impact how well a model performs. One of the biggest being the hyperparameters, even small changes (like bumping n_neighbors from 5 to 6 in KNN) completely changed the results in my case. That’s why tuning is so important, especially on small datasets.

The size of the dataset also plays a role. The Iris dataset is super clean, well-balanced, and not very large, which makes it easy for most classifiers to do well. On a bigger or messier dataset, the same models might behave really differently.

Scaling features is another factor — especially for models like KNN and SVM that rely on distance calculations. Without scaling, they’d probably perform a lot worse.

And of course, the model itself matters. Different classifiers are better suited to different types of problems. In this case, they all worked well, but that might not be the case with more complex datasets!

## Acknowledgments
- Iris Flowers (Dataset Utilized) - [Kaggle.com](https://www.kaggle.com/datasets/arshid/iris-flower-dataset?resource=download)

- ChatGPT (Bug Resolution) - [OpenAI](https://chatgpt.com)

- Stack Overflow (Library Usage & Syntax) - [StackOverflow](https://stackoverflow.com/questions)