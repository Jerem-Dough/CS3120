## Introduction
This model predicts the likelihood of diabetes using the publicly available Pima Indians Diabetes Database. The code incorporates SKLearn, NumPy, and Matplotlib, with model training performed using logistic regression.

## Data Input
Data was split into two groups, 60% allocated for training and 40% allocated for testing. Prior to utilization of data, NaN and misrepresented values were replaced by the median of their feature. Features selected for usage included pregnant, glucose, bmi, pedigree, and age.

## Model & Metrics
This implimentation is a Logistic Regression model for Binary Classification. To determine model performance, values such as precision, recall, F score, and AUC are calculated. The model performs well, not quite excellent, with an AUC value equal to 0.821.

## Acknowledgments
- Pima Indians Diabetes Database (Dataset Utilized) - [Machine Learning Mastery](https://machinelearningmastery.com/standard-machine-learning-datasets/)

- ChatGPT (Bug Resolution) - [OpenAI](https://chatgpt.com)

- Stack Overflow (Library Usage & Syntax) - [StackOverflow](https://stackoverflow.com/questions)