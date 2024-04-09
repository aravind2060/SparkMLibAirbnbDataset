from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Function to create a Spark session
def create_spark_session():
    spark = SparkSession.builder.appName("RandomForestExample").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Function to load and show data
def load_and_show_data(spark, file_path):
    df = spark.read.parquet(file_path)
    df.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms", "number_of_reviews", "price").show(5)
    return df

# Function to split data into training and testing sets
def split_data(df):
    train_df, test_df = df.randomSplit([.8, .2], seed=42)
    print(f"There are {train_df.count()} rows in the training set, and {test_df.count()} in the test set")
    return train_df, test_df

# Function to define and fit the model
def define_and_fit_model(train_df):
    # Define the Random Forest Regressor
    rf = RandomForestRegressor(labelCol="price", maxBins=40)

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

    # Fit the model
    pipeline_model = pipeline.fit(train_df)
    return pipeline_model

# Function to evaluate the model and save predictions to CSV
def evaluate_model_and_save_predictions(pipeline_model, test_df, output_path):
    pred_df = pipeline_model.transform(test_df)
    pred_df.select("price", "prediction").show()

    # Evaluate the model
    regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regression_evaluator.evaluate(pred_df)
    print(f"RMSE is {rmse:.1f}")

    # Save the prediction results to a CSV file
    pred_df.select("price", "prediction").write.csv(output_path, header=True)

# Main function to orchestrate the model building and evaluation
def main():
    file_path = "/workspaces/SparkMLibAirbnbDataset/data/input/sf-airbnb-clean.parquet"
    output_path = "/workspaces/SparkMLibAirbnbDataset/data/output/Random/predictions.csv"
    spark = create_spark_session()
    df = load_and_show_data(spark, file_path)
    train_df, test_df = split_data(df)
    pipeline_model = define_and_fit_model(train_df)
    evaluate_model_and_save_predictions(pipeline_model, test_df, output_path)

# Run the main function
if __name__ == "__main__":
    main()
