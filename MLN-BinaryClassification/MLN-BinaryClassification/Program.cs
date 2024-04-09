using System;

namespace MLN_BinaryClassification
{
    class Program
    {
        static string cr = Environment.NewLine;

        static void Main(string[] args)
        {
            var ml = new ML();
            ml.Seup(@"..\..\..\Data\titanic.mdl");

            ml.MakePrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 60.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();
            ml.MakePrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 60.0f, Gender = "male", FareClass = 3.0f });
            Console.WriteLine();
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "female", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "female", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "male", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "male", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 20.0f, Gender = "male", FareClass = 3.0f });
            Console.WriteLine();
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "female", FareClass = 3.0f });
            Console.WriteLine();
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 1.0f });
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 2.0f });
            ml.MakePrediction(new Input { Age = 12.0f, Gender = "male", FareClass = 3.0f });
            
            Console.WriteLine($"{cr}{cr}");
        }
    }
}
