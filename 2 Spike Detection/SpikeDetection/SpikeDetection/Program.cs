using Microsoft.ML;
using SpikeDetection;

string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "traffic.csv");
//assign the Number of records in dataset file to constant variable
const int _docsize = 36;

MLContext mlContext = new MLContext();

IDataView dataView = mlContext.Data.LoadFromTextFile<TrafficData>(path: _dataPath, hasHeader: true, separatorChar: ',');


//DetectSpike(mlContext, _docsize, dataView);

DetectChangepoint(mlContext, _docsize, dataView);


static void DetectSpike(MLContext mlContext, int docSize, IDataView trafficData)
{
	Console.WriteLine("Detect temporary changes in pattern");

	// STEP 2: Set the training algorithm
	var iidSpikeEstimator = mlContext.Transforms.DetectIidSpike(outputColumnName: nameof(TrafficPrediction.Prediction), inputColumnName: nameof(TrafficData.TrafficVolume), confidence: 95.0, pvalueHistoryLength: docSize / 4);

	// STEP 3: Create the transform
	// Create the spike detection transform
	Console.WriteLine("=============== Training the model ===============");
	ITransformer iidSpikeTransform = iidSpikeEstimator.Fit(CreateEmptyDataView(mlContext));

	Console.WriteLine("=============== End of training process ===============");

	//Apply data transformation to create predictions.
	IDataView transformedData = iidSpikeTransform.Transform(trafficData);

	var predictions = mlContext.Data.CreateEnumerable<TrafficPrediction>(transformedData, reuseRowObject: false);

	Console.WriteLine("Alert\tScore\tP-Value");

	foreach (var p in predictions)
	{
		var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}";

		if (p.Prediction[0] == 1)
		{
			results += " <-- Spike detected";
		}

		Console.WriteLine(results);
	}
	Console.WriteLine("");
}

static void DetectChangepoint(MLContext mlContext, int docSize, IDataView productSales)
{
	Console.WriteLine("Detect Persistent changes in pattern");

	//STEP 2: Set the training algorithm
	var iidChangePointEstimator = mlContext.Transforms.DetectIidChangePoint(outputColumnName: nameof(TrafficPrediction.Prediction), inputColumnName: nameof(TrafficData.TrafficVolume), confidence: 95.0, changeHistoryLength: docSize / 4);

	//STEP 3: Create the transform
	Console.WriteLine("=============== Training the model Using Change Point Detection Algorithm===============");
	var iidChangePointTransform = iidChangePointEstimator.Fit(CreateEmptyDataView(mlContext));
	Console.WriteLine("=============== End of training process ===============");

	//Apply data transformation to create predictions.
	IDataView transformedData = iidChangePointTransform.Transform(productSales);
	var predictions = mlContext.Data.CreateEnumerable<TrafficPrediction>(transformedData, reuseRowObject: false);
	Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
	foreach (var p in predictions)
	{
		var results = $"{p.Prediction[0]}\t{p.Prediction[1]:f2}\t{p.Prediction[2]:F2}\t{p.Prediction[3]:F2}";

		if (p.Prediction[0] == 1)
		{
			results += " <-- alert is on, predicted changepoint";
		}
		Console.WriteLine(results);
	}
	Console.WriteLine("");
}

static IDataView CreateEmptyDataView(MLContext mlContext)
{
	// Create empty DataView. We just need the schema to call Fit() for the time series transforms
	IEnumerable<TrafficData> enumerableData = new List<TrafficData>();
	return mlContext.Data.LoadFromEnumerable(enumerableData);
}