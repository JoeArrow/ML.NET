﻿using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

using System;
using System.IO;
using System.Linq;

namespace MLN_BinaryClassification
{
    class Program
    {
        static readonly string _path = "..\\..\\..\\Data\\titanic.csv";
        static readonly string _model = "..\\..\\..\\Data\\titanic.mdl";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);

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
            var testData = trainTestData.TestSet;

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

            if(!File.Exists(_model))
            {
                //context.Data.Save();
            }
                
            var model = pipeline.Fit(trainData);

            // ------------------
            // Evaluate the model

            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine();
            Console.WriteLine($"Accuracy: {metrics.Accuracy:P1}");
            Console.WriteLine($"AUC: {metrics.AreaUnderPrecisionRecallCurve:P1}");
            Console.WriteLine($"F1: {metrics.F1Score:P1}");
            Console.WriteLine();

            // ---------------------------------
            // Use the model to make predictions

            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            var input = new Input { Age = 60.0f, Gender = "female", FareClass = 1.0f };
            var output = predictor.Predict(input);

            Console.WriteLine();
            Console.WriteLine($"Probability that a {input.Age}-year old {input.Gender} traveling in first class will survive: {output.Probability:P1}");

            input = new Input { Age = 60.0f, Gender = "male", FareClass = 3.0f };
            output = predictor.Predict(input);

            Console.WriteLine($"Probability that a {input.Age}-year old {input.Gender} traveling in third class will survive: {output.Probability:P1}");
        }
    }
}
