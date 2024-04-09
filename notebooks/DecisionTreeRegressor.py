from pyspark.sql import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize SparkSession
def create_spark_session():
    """
    Create and configure the Spark session.
    Returns the SparkSession object.
    """
    spark = SparkSession.builder.appName("DecisionTrees").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

# Load data
def load_data(spark, file_path):
    """
    Load data from the specified file path using the given Spark session.
    Returns the DataFrame loaded with the data.
    """
    return spark.read.parquet(file_path)

# Show and save initial data
def show_and_save_initial_data(df, output_path):
    """
    Display and save the initial data from the DataFrame to a CSV file.
    """
    df.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms", "number_of_reviews", "price").show(5)
    df.write.option("header", True).csv(output_path)

# Prepare data for model
def prepare_data(df):
    """
    Split the dataset into training and testing sets.
    Returns the training and testing DataFrames.
    """
    return df.randomSplit([0.8, 0.2], seed=42)

# Build and train the decision tree model
def train_decision_tree(trainDF):
    """
    Train a Decision Tree regressor using the training data.
    Returns the trained model and the pipeline.
    """
    dt = DecisionTreeRegressor(labelCol="price" ,maxBins=40)
    categoricalCols = [field for field, dataType in trainDF.dtypes if dataType == "string"]
    indexOutputCols = [x + "Index" for x in categoricalCols]
    stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")
    numericCols = [field for field, dataType in trainDF.dtypes if dataType == "double" and field != "price"]
    assemblerInputs = indexOutputCols + numericCols
    vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    pipeline = Pipeline(stages=[stringIndexer, vecAssembler, dt])
    model = pipeline.fit(trainDF)
    return model

# Evaluate and show feature importances
def evaluate_model(spark,model, vecAssembler, output_path):
    """
    Evaluate the model and show feature importances.
    Save the feature importances to a CSV file.
    """
    dtModel = model.stages[-1]
    print(dtModel.toDebugString)
    feature_importances_float = [float(val) for val in dtModel.featureImportances]
    schema = "feature STRING, importance FLOAT"
    featureImp = spark.createDataFrame(zip(vecAssembler.getInputCols(), feature_importances_float), schema)
    featureImp.orderBy("importance", ascending=False).show()
    featureImp.write.option("header", True).csv(output_path)

# Generate predictions and evaluate the model
def generate_predictions(model, testDF, predictions_output, spark):
    """
    Use the model to make predictions on the test set and evaluate the model's performance.
    Save the predictions to a CSV file and print the RMSE.
    """
    predDF = model.transform(testDF)
    # Select only the necessary columns, excluding 'features'
    predDF = predDF.select("price", "prediction")
    predDF.show()
    
    # Save the predictions as CSV
    predDF.write.option("header", True).csv(predictions_output)
    
    # Evaluate model performance
    regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
    rmse = regressionEvaluator.evaluate(predDF)
    print(f"RMSE is {rmse:.1f}")


# Main function to orchestrate the operations
def main():
    spark = create_spark_session()
    file_path = "/workspaces/SparkMLibAirbnbDataset/data/input/sf-airbnb-clean.parquet"
    airbnbDF = load_data(spark, file_path)
    initial_data_output = "/workspaces/SparkMLibAirbnbDataset/data/output/initial_data.csv"
    show_and_save_initial_data(airbnbDF, initial_data_output)
    trainDF, testDF = prepare_data(airbnbDF)
    model = train_decision_tree(trainDF)
    feature_importance_output = "/workspaces/SparkMLibAirbnbDataset/data/output/feature_importances.csv"
    vecAssembler = model.stages[1]  # Assuming VectorAssembler is the second stage in the pipeline
    evaluate_model(spark,model, vecAssembler, feature_importance_output)
    predictions_output = "/workspaces/SparkMLibAirbnbDataset/data/output/predictions.csv"
    generate_predictions(model, testDF, predictions_output, spark)

# Entry point of the script
if __name__ == "__main__":
    main()
