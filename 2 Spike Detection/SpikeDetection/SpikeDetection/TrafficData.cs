using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SpikeDetection
{
	internal class TrafficData
	{
		[LoadColumn(0)]
		public string? Month { get; set; }

		[LoadColumn(1)]
		public float TrafficVolume { get; set; }
	}
}
