﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace DigitClassification
{
    class Program
    {
        static readonly string cr = Environment.NewLine;
        static readonly string _trainDataPath = "..\\..\\..\\Data\\mnist-digits-train.csv";
        static readonly string _testDataPath = "..\\..\\..\\Data\\mnist-digits-test.csv";

        static void Main(string[] args)
        {
            var context = new MLContext(seed: 0);
            var trainData = context.Data.LoadFromTextFile<Input>(_trainDataPath, hasHeader: false, separatorChar: ',');
            var testData = context.Data.LoadFromTextFile<Input>(_testDataPath, hasHeader: false, separatorChar: ',');

            // Build and train the model

            var pipeline = context.Transforms.Conversion.MapValueToKey("Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(context.Transforms.Concatenate("Features", "PixelValues"))
                .Append(context.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training the model...");
            var model = pipeline.Fit(trainData);

            // Evaluate the model

            var predictions = model.Transform(testData);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"{cr}Macro accuracy = {metrics.MacroAccuracy:P2}");
            Console.WriteLine($"Micro accuracy = {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}{cr}");

            // Use the model to make a prediction

            var predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            var input = new Input
            {
                PixelValues = new float[]
                {
                    0, 0,  1,  0, 12,  2, 0, 0,
                    0, 0,  0,  6, 14,  1, 0, 0,
                    0, 0,  4, 16,  7,  8, 0, 0,
                    0, 0, 13, 10,  0, 16, 6, 0,
                    0, 3, 16, 10, 12, 16, 0, 0,
                    0, 0,  4, 10, 13, 16, 0, 0,
                    0, 0,  0,  0,  6, 16, 0, 0,
                    1, 0,  0,  0, 12,  8, 0, 0
                }
            }; // 4

            var prediction = predictor.Predict(input);

            int i = 0;

            foreach (var score in prediction.Scores)
            {
                Console.WriteLine($"{i++} - {score:N8}");
            }

            Console.WriteLine();
            Console.WriteLine($"Looks like a {prediction.Digit}!");
            Console.WriteLine();
        }
    }

    class Input
    {
        [LoadColumn(0, 63), VectorType(64)]
        public float[] PixelValues;

        [LoadColumn(64), ColumnName("Label")]
        public int Digit;
    }

    class Output
    {
        [ColumnName("Score")]
        public float[] Scores;

        [ColumnName("PredictedLabel")]
        public int Digit;
    }
}