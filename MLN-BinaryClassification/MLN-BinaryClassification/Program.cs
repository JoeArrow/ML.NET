
using System;
using System.Linq;

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MLN_BinaryClassification
{
    class Program
    {
        static readonly string _path = @".\Data\titanic.csv";
        private static readonly string cr = Environment.NewLine;
        private static PredictionEngine<Input, Output> _predictor;

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
            var model = pipeline.Fit(trainData);

            // ------------------
            // Evaluate the model

            var predictions = model.Transform(testData);
            var metrics = context.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine($"{cr}Accuracy: {metrics.Accuracy:P1}");
            Console.WriteLine($"AUC:\t{metrics.AreaUnderPrecisionRecallCurve:P1}");
            Console.WriteLine($"F1:\t{metrics.F1Score:P1}{cr}");

            // ---------------------------------
            // Use the model to make predictions

            _predictor = context.Model.CreatePredictionEngine<Input, Output>(model);

            ShowPrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();
            
            ShowPrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 3.0f });
            Console.WriteLine();
            
            ShowPrediction(new Input { Age = 30.0f, Gender = "female", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 30.0f, Gender = "female", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 30.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();
            
            ShowPrediction(new Input { Age = 30.0f, Gender = "male", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 30.0f, Gender = "male", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 30.0f, Gender = "male", FareClass = 3.0f });
            Console.WriteLine();
            
            ShowPrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();

            ShowPrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 1.0f });
            ShowPrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 2.0f });
            ShowPrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 3.0f });
        }

        // ------------------------------------------------

        public static void ShowPrediction(Input input)
        {
            var output = _predictor.Predict(input);
            Console.WriteLine($"Probability that a {input.Age}-year old {input.Gender} traveling in {Input.GetFareClass(input.FareClass)} will survive: {output.Probability:P1}");
        }
    }

    // ----------------------------------------------------

    public class Input
    {
        [LoadColumn(5)]
        public float Age;

        [LoadColumn(4)]
        public string Gender;

        [LoadColumn(2)]
        public float FareClass;

        [LoadColumn(1), ColumnName("Label")]
        public bool Survived;

        // ------------------------------------------------

        public static string GetFareClass(float fareClass) 
        {
            var retVal = "Unknown Class";

            if(fareClass == 1.0f)
            {
                retVal = "First Class";
            }
            else if(fareClass == 2.0f)
            {
                retVal = "Second Class";
            }
            else if(fareClass == 3.0f)
            {
                retVal = "Steerage";
            }

            return retVal;
        }
    }

    // ----------------------------------------------------

    public class Output
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }
        public float Probability { get; set; }
    }
}
