﻿using Microsoft.ML.Data;

namespace AutoML
{
    public class Input
    {
        [LoadColumn(1)]
        public float Bathrooms;

        [LoadColumn(2)]
        public float Bedrooms;

        [LoadColumn(3)]
        public float FinishedSquareFeet;

        [LoadColumn(5), ColumnName("Label")]
        public float LastSoldPrice;

        [LoadColumn(9)]
        public float TotalRooms;

        [LoadColumn(10)]
        public string UseCode;
    }
}
