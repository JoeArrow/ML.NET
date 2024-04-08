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

        // ------------------------------------------------

        public string GetFareClass()
        {
            var retVal = "Unknown Class";

            switch(FareClass)
            {
                case 1.0f:
                    retVal = "First Class";
                    break;

                case 2.0f:
                    retVal = "Second Class";
                    break;

                case 3.0f:
                    retVal = "Steerage";
                    break;
            }

            return retVal;
        }
    }
}
