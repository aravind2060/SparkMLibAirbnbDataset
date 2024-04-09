from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Function to create a Spark session
def create_spark_session():
    spark = SparkSession.builder.appName("RandomForestCVExample").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Function to load data
def load_data(spark, file_path):
    df = spark.read.parquet(file_path)
    df.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms", "number_of_reviews", "price").show(5)
    return df

# Function to split data into training and testing sets
def split_data(df):
    train_df, test_df = df.randomSplit([.8, .2], seed=42)
    return train_df, test_df

# Function to perform k-fold cross-validation and fit the model
def cross_validate_and_fit_model(train_df):
    # Define the Random Forest Regressor
    rf = RandomForestRegressor(labelCol="price")

    # Handle categorical features
    categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
    index_output_cols = [x + "Index" for x in categorical_cols]
    string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

    # Handle numeric features
    numeric_cols = [field for (field, dataType) in train_df.dtypes if dataType == "double" and field != "price"]
    assembler_inputs = index_output_cols + numeric_cols
    vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # Create a Pipeline
    stages = [string_indexer, vec_assembler, rf]
    pipeline = Pipeline(stages=stages)

    # Define the parameter grid
    param_grid = (ParamGridBuilder()
              .addGrid(rf.maxDepth, [2, 4, 6])  # Example depths, adjust as necessary
              .addGrid(rf.numTrees, [10, 100])  # Example number of trees, adjust as necessary
              .addGrid(rf.maxBins, [40, 50, 60])  # Increase maxBins to accommodate the number of categories
              .build())

    # Define the evaluator
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

    # Set up cross-validation
    cv = CrossValidator(estimator=pipeline,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=3,
                        seed=42)

    # Fit the model
    cv_model = cv.fit(train_df)
    return cv_model

# Function to apply the best model and get top 20 predictions
def apply_model_and_get_top_predictions(cv_model, test_df):
    pred_df = cv_model.transform(test_df)
    top_pred_df = pred_df.orderBy("prediction", ascending=False).limit(20)
    top_pred_df.show()

    # Calculate and print RMSE
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = evaluator.evaluate(pred_df)
    print(f"RMSE is {rmse:.1f}")

    # Select necessary columns before saving to CSV
    selected_columns = [col for col in pred_df.columns if col != 'features']
    top_pred_df.select(*selected_columns).write.csv("/workspaces/SparkMLibAirbnbDataset/data/output/Kflod/top_predictions.csv", header=True)

    return top_pred_df


# Main function to orchestrate the model building and evaluation
def main():
    file_path = "/workspaces/SparkMLibAirbnbDataset/data/input/sf-airbnb-clean.parquet"
    spark = create_spark_session()
    df = load_data(spark, file_path)
    train_df, test_df = split_data(df)
    cv_model = cross_validate_and_fit_model(train_df)
    top_pred_df = apply_model_and_get_top_predictions(cv_model, test_df)
    # Optionally, save the top predictions to a CSV
    top_pred_df.write.csv("/workspaces/SparkMLibAirbnbDataset/data/output/Kflod/top_predictions.csv", header=True)

# Run the main function
if __name__ == "__main__":
    main()
