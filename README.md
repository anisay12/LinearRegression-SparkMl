# USA Housing Price Prediction using PySpark

This project demonstrates the process of predicting housing prices in the USA using a Linear Regression model implemented in PySpark. The dataset used for this project is `USA_Housing.csv`.

## Prerequisites

Ensure you have the following installed:
- Apache Spark
- Python (version 3.6 or later)
- PySpark library
- JDK (Java Development Kit)

## Setup

1. **Setting Up the Environment:**
   - Change the working directory to the desired location where your dataset is stored.

2. **Loading Data:**
   - Read the dataset using SparkContext and cache it for better performance.

3. **Data Preparation:**
   - Remove the header and filter out any rows that contain non-numeric values.
   - Convert the cleaned data into feature vectors suitable for statistical analysis.

4. **Statistical Analysis:**
   - Compute basic statistics such as mean, variance, and correlation of the features.

5. **DataFrame Creation:**
   - Transform the RDD into a DataFrame with labeled points, splitting the data into training and testing sets.

6. **Model Training:**
   - Train a Linear Regression model on the training dataset.

7. **Model Evaluation:**
   - Evaluate the model using various regression metrics such as R², RMSE, and MSE.

## Execution Steps

1. **Change Directory:**
   ```python
   import os
   os.chdir("./workspace")
   ```

2. **Load Dataset:**
   ```python
   autoData = sc.textFile("./USA_Housing.csv")
   autoData.cache()
   ```

3. **Data Cleaning and Transformation:**
   ```python
   firstLine = autoData.first()
   dataLines = autoData.filter(lambda x: x != firstLine)
   import string
   ALPHA = string.ascii_letters
   filteredRDD = autoData.filter(lambda x: not x.startswith(tuple(ALPHA)))

   from pyspark.mllib.linalg import Vectors
   def nettoyage(input):
       featurelist = input.split(",")
       valeurs = Vectors.dense([float(featurelist[i]) for i in range(6)])
       return valeurs

   autoVectors = filteredRDD.map(nettoyage)
   ```

4. **Statistical Analysis:**
   ```python
   from pyspark.mllib.stat import Statistics
   autostats = Statistics.colStats(autoVectors)
   ```

5. **DataFrame Creation:**
   ```python
   from pyspark.sql import SQLContext
   sqlContext = SQLContext(sc)

   def splitdata(input):
       lp = (float(input[5]), Vectors.dense([input[i] for i in range(5)]))
       return lp

   autolp = autoVectors.map(splitdata)
   dataframe = sqlContext.createDataFrame(autolp, ["label", "features"])
   (trainingData, testData) = dataframe.randomSplit([0.9, 0.1])
   ```

6. **Model Training and Prediction:**
   ```python
   from pyspark.ml.regression import LinearRegression
   lr = LinearRegression(maxIter=10)
   from pyspark.mllib.util import MLUtils
   Convertedtrainingvecdf = MLUtils.convertVectorColumnsToML(trainingData)
   lrModel = lr.fit(Convertedtrainingvecdf)
   convertedtesttodf = MLUtils.convertVectorColumnsToML(testData)
   prediction = lrModel.transform(convertedtesttodf)
   ```

7. **Model Evaluation:**
   ```python
   from pyspark.ml.evaluation import RegressionEvaluator
   evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="r2")
   r2 = evaluator.evaluate(prediction)
   evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
   rmse = evaluator.evaluate(prediction)
   evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="mse")
   mse = evaluator.evaluate(prediction)
   ```

## Results
- **Model Coefficients:**
  ```python
  print("Coefficients: " + str(lrModel.coefficients))
  print("Intercept: " + str(lrModel.intercept))
  ```
- **Performance Metrics:**
  - R²
  - RMSE
  - MSE

## Conclusion

This project provides a comprehensive pipeline for data preprocessing, statistical analysis, model training, and evaluation using PySpark. The results offer insights into the predictive power of the linear regression model for housing prices.
