using Microsoft.ML.Data;

namespace MLN_BinaryClassification
{
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
    }
}
