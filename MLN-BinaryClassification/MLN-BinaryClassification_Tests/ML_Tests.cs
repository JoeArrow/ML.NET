using MLN_BinaryClassification;

using System.Text.Json;

namespace MLN_BinaryClassification_Tests
{
    [TestClass]
    public class ML_Tests
    {
        [TestMethod]
        [DataRow(@".\Data\titanic.mdl", 60.0f, "female", 1.0f, 0.92f)]
        [DataRow(@".\Data\titanic.mdl", 60.0f, "female", 2.0f, 0.77f)]
        [DataRow(@".\Data\titanic.mdl", 60.0f, "female", 3.0f, 0.20f)]

        [DataRow(@".\Data\titanic.mdl", 60.0f, "male", 1.0f, 0.10f)]
        [DataRow(@".\Data\titanic.mdl", 60.0f, "male", 2.0f, 0.03f)]
        [DataRow(@".\Data\titanic.mdl", 60.0f, "male", 3.0f, 0.025f)]

        [DataRow(@".\Data\titanic.mdl", 20.0f, "female", 1.0f, 0.97f)]
        [DataRow(@".\Data\titanic.mdl", 20.0f, "female", 2.0f, 0.98f)]
        [DataRow(@".\Data\titanic.mdl", 20.0f, "female", 3.0f, 0.39f)]

        [DataRow(@".\Data\titanic.mdl", 20.0f, "male", 1.0f, 0.12f)]
        [DataRow(@".\Data\titanic.mdl", 20.0f, "male", 2.0f, 0.12f)]
        [DataRow(@".\Data\titanic.mdl", 20.0f, "male", 3.0f, 0.18f)]
        public void MakePrediction_ML(string modelPath, float age, string gender, float fareClass, float expected)
        {
            // -------
            // Arrange

            var sut = new ML();
            var input = new Input() { Age = age, Gender = gender, FareClass = fareClass };

            sut.Seup(modelPath);

            // ---
            // Act

            var resp = sut.MakePrediction(input);

            // ---
            // Log

            Console.WriteLine($"{resp.Prediction} - {resp.Probability}");

            // ------
            // Assert

            Assert.IsTrue(resp.Probability >= expected);
        }
    }
}