using System;
using System.IO;

using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace MLN_BinaryClassification
{
    public class ML
    {
        private IDataView testData;
        private ITransformer model = null;
        private readonly string cr = Environment.NewLine;
        private PredictionEngine<Input, Output> predictor;
        private MLContext context = new MLContext(seed: 0);
        private readonly string _path = "..\\..\\..\\Data\\titanic.csv";
        private readonly string _modelPath = "..\\..\\..\\Data\\titanic.mdl";

        // ------------------------------------------------

        public void Seup()
        {
            if(File.Exists(_modelPath))
            {
                Console.WriteLine($"{cr}Model pre-trained...{cr}");
                model = context.Model.Load(_modelPath, out _);
            }
            else
            {
                // -------------
                // Load the data

                var data = context.Data.LoadFromTextFile<Input>(_path, hasHeader: true, allowQuoting: true, separatorChar: ',');

                // ---------------------------------------------------------------------
                // Uncomment the following line to remove rows with missing "Age" values

                //data = context.Data.FilterRowsByMissingValues(data, "Age");

                // -------------------------------------------------
                // Split the data into a training set and a test set

                var trainTestData = context.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
                var trainData = trainTestData.TrainSet;
                testData = trainTestData.TestSet;

                // -----------------------------------------------------------------------
                // Build and train the model, replacing missing values in the "Age" column
                // with the mean of all "Age" values, normalizing the resulting "Age" values,
                // and one-hot encoding the "Gender" and "FareClass" columns

                var pipeline = context.Transforms.ReplaceMissingValues("Age", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                    .Append(context.Transforms.NormalizeMeanVariance("Age"))
                    .Append(context.Transforms.Categorical.OneHotEncoding(inputColumnName: "Gender", outputColumnName: "GenderEncoded"))
                    .Append(context.Transforms.Categorical.OneHotEncoding(inputColumnName: "FareClass", outputColumnName: "FareClassEncoded"))
                    .Append(context.Transforms.Concatenate("Features", "Age", "GenderEncoded", "FareClassEncoded"))
                    .Append(context.BinaryClassification.Trainers.LightGbm());

                Console.WriteLine("Training the model...");

                model = pipeline.Fit(trainData);

                if(!File.Exists(_modelPath))
                {
                    context.Model.Save(model, trainData.Schema, _modelPath);
                }

                Evaluate();
            }

            // ---------------------------------
            // Use the model to make predictions

            predictor = context.Model.CreatePredictionEngine<Input, Output>(model);
        }

        // ------------------------------------------------

        private void Evaluate()
        {
            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine($"{cr}Accuracy: {metrics.Accuracy:P1}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P1}");
            Console.WriteLine($"F1: {metrics.F1Score:P1}{cr}");
        }

        // ------------------------------------------------

        public void MakePrediction(Input input)
        {
            var output = predictor.Predict(input);
            Console.WriteLine($"Probability that a {input.Age}-year old {input.Gender} traveling in {input.GetFareClass()} would survive:\t{output.Probability:P1}");
        }
    }
}
