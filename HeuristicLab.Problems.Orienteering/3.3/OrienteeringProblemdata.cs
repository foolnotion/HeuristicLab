﻿using System;
using HEAL.Attic;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.IntegerVectorEncoding;
using HeuristicLab.Problems.Instances.Types;
using HeuristicLab.Problems.TravelingSalesman;

namespace HeuristicLab.Problems.Orienteering {
  [StorableType("c33063a6-a2ee-4054-9dd8-46a738003139")]
  public interface IOrienteeringProblemData : INamedItem {
    ITSPData RoutingData { get; }
    int StartingPoint { get; }
    int TerminalPoint { get; }
    double MaximumTravelCosts { get; }
    double PointVisitingCosts { get; }

    double GetScore(int city);
    OrienteeringSolution GetSolution(IntegerVector route, double quality, double score, double travelCosts);
    OPData Export();
  }

  [Item("Orienteering Problem Data", "Represents the main data for an orienteering problem.")]
  [StorableType("d7d2d61a-7d51-4254-ab68-375942106058")]
  public class OrienteeringProblemData : NamedItem, IOrienteeringProblemData {
    [Storable] public ITSPData RoutingData { get; protected set; }
    [Storable] public int StartingPoint { get; protected set; }
    [Storable] public int TerminalPoint { get; protected set; }
    [Storable] public DoubleArray Scores { get; protected set; }
    [Storable] public double MaximumTravelCosts { get; protected set; }
    [Storable] public double PointVisitingCosts { get; protected set; }

    [StorableConstructor]
    protected OrienteeringProblemData(StorableConstructorFlag _) : base(_) { }
    protected OrienteeringProblemData(OrienteeringProblemData original, Cloner cloner) : base(original, cloner) {
      RoutingData = original.RoutingData;
      StartingPoint = original.StartingPoint;
      TerminalPoint = original.TerminalPoint;
      Scores = original.Scores;
      MaximumTravelCosts = original.MaximumTravelCosts;
      PointVisitingCosts = original.PointVisitingCosts;
    }
    public OrienteeringProblemData() {
      RoutingData = new MatrixTSPData("HL OP Default", defaultMatrix, defaultCoordinates);
      Name = RoutingData.Name;
      StartingPoint = 0;
      TerminalPoint = 20;
      Scores = new DoubleArray(defaultScores, @readonly: true);
      MaximumTravelCosts = 30;
      PointVisitingCosts = 0;
    }
    public OrienteeringProblemData(ITSPData tspData, int startingPoint, int terminalPoint, double[] scores, double maxDist, double pointVisitCosts)
      : this(tspData, startingPoint, terminalPoint, new DoubleArray(scores, @readonly: true), maxDist, pointVisitCosts) { }
    public OrienteeringProblemData(ITSPData tspData, int startingPoint, int terminalPoint, DoubleArray scores, double maxDist, double pointVisitCosts)
      : base(tspData.Name, tspData.Description) {
      if (tspData.Cities != scores.Length) throw new ArgumentException("Unequal number of cities and scores.");
      if (startingPoint < 0 || startingPoint >= tspData.Cities) throw new ArgumentException("Starting point is not in the range of cities.", "startingPoint");
      if (terminalPoint < 0 || terminalPoint >= tspData.Cities) throw new ArgumentException("Terminal point is not in the range of cities.", "terminalPoint");
      RoutingData = tspData;
      StartingPoint = startingPoint;
      TerminalPoint = terminalPoint;
      Scores = scores.AsReadOnly();
      MaximumTravelCosts = maxDist;
      PointVisitingCosts = pointVisitCosts;
    }

    public override IDeepCloneable Clone(Cloner cloner) {
      return new OrienteeringProblemData(this, cloner);
    }

    public double GetScore(int site) => Scores[site];

    public OrienteeringSolution GetSolution(IntegerVector route, double quality, double score, double travelCosts) {
      return new OrienteeringSolution(route, this, quality, score, travelCosts);
    }

    public OPData Export() {
      var tspExport = RoutingData.Export();
      return new OPData() {
        Name = tspExport.Name,
        Description = tspExport.Description,
        Coordinates = tspExport.Coordinates,
        Dimension = tspExport.Dimension,
        DistanceMeasure = tspExport.DistanceMeasure,
        Distances = tspExport.Distances,
        MaximumDistance = MaximumTravelCosts,
        Scores = Scores.CloneAsArray(),
        StartingPoint = StartingPoint,
        TerminalPoint = TerminalPoint,
        PointVisitingCosts = PointVisitingCosts
      };
    }

    private static double[,] defaultCoordinates = new double[21, 2] {
      {  4.60,  7.10 }, {  5.70, 11.40 }, {  4.40, 12.30 }, {  2.80, 14.30 }, {  3.20, 10.30 },
      {  3.50,  9.80 }, {  4.40,  8.40 }, {  7.80, 11.00 }, {  8.80,  9.80 }, {  7.70,  8.20 },
      {  6.30,  7.90 }, {  5.40,  8.20 }, {  5.80,  6.80 }, {  6.70,  5.80 }, { 13.80, 13.10 },
      { 14.10, 14.20 }, { 11.20, 13.60 }, {  9.70, 16.40 }, {  9.50, 18.80 }, {  4.70, 16.80 },
      {  5.00,  5.60 }
    };
    private static double[] defaultScores = new double[21] { 0, 20, 20, 30, 15, 15, 10, 20, 20, 20, 15, 10, 10, 25, 40, 40, 30, 30, 50, 30, 0 };
    private static double[,] defaultMatrix = new double[21, 21] {
      { 0, 4.43846820423443, 5.2038447325030761, 7.4215901261117905, 3.492849839314597, 2.9154759474226513, 1.3152946437965911, 5.0447993022517759, 4.9929950931279725, 3.2893768406797057, 1.878829422805594, 1.3601470508735445, 1.2369316876852983, 2.4698178070456942, 10.983624174196786, 11.860016863394419, 9.2633687176966024, 10.606601717798211, 12.684636376341263, 9.7005154502222215, 1.5524174696260025 },
      { 4.43846820423443, 0, 1.5811388300841898, 4.1012193308819764, 2.7313000567495322, 2.7202941017470885, 3.2695565448543631, 2.1377558326431947, 3.4885527085024819, 3.7735924528226423, 3.5510561809129406, 3.2140317359976405, 4.6010868281309367, 5.6885850613311577, 8.2764726786234259, 8.85437744847146, 5.9236812878479537, 6.4031242374328468, 8.3186537372341682, 5.4918120870983929, 5.8420886675914128 },
      { 5.2038447325030761, 1.5811388300841898, 0, 2.56124969497314, 2.3323807579381204, 2.6570660511172846, 3.9000000000000004, 3.640054944640259, 5.0606323715519981, 5.2630789467763082, 4.7927027865287037, 4.22018956920184, 5.6753854494650851, 6.89492567037528, 9.4339811320566049, 9.8843310345212529, 6.92314957226839, 6.700746227100379, 8.2619610262939389, 4.5099889135118723, 6.7268120235368558 },
      { 7.4215901261117905, 4.1012193308819764, 2.56124969497314, 0, 4.0199502484483558, 4.5541190146942805, 6.11310068623117, 5.990826320300064, 7.5000000000000009, 7.8243210568074222, 7.2945184899347542, 6.6309878600401628, 8.0777472107017569, 9.35200513259055, 11.065260954898443, 11.300442469213319, 8.4291162051546049, 7.2124891681027838, 8.0709355095924291, 3.1400636936215167, 8.9738509013689338 },
      { 3.492849839314597, 2.7313000567495322, 2.3323807579381204, 4.0199502484483558, 0, 0.58309518948453, 2.2472205054244236, 4.6529560496527358, 5.6222771187482392, 4.9658836071740557, 3.9204591567825315, 3.041381265149111, 4.360045871318329, 5.7008771254956905, 10.963576058932597, 11.576700738984314, 8.6539008545279739, 8.9140338792266185, 10.580170130957253, 6.6708320320631671, 5.0328918128646487 },
      { 2.9154759474226513, 2.7202941017470885, 2.6570660511172846, 4.5541190146942805, 0.58309518948453, 0, 1.6643316977093243, 4.4643028571099421, 5.3000000000000007, 4.4944410108488473, 3.3837848631377261, 2.4839484696748451, 3.7802116342871606, 5.12249938994628, 10.815729286552989, 11.476933388322857, 8.5866174946832228, 9.0553851381374155, 10.816653826391969, 7.1021123618258812, 4.4598206241955527 },
      { 1.3152946437965911, 3.2695565448543631, 3.9000000000000004, 6.11310068623117, 2.2472205054244236, 1.6643316977093243, 0, 4.2801869118065383, 4.6173585522460785, 3.306055050963308, 1.9646882704388495, 1.0198039027185573, 2.12602916254693, 3.4713109915419564, 10.509519494249012, 11.301769772916098, 8.5603738236130766, 9.5963534741067118, 11.583177456984764, 8.40535543567314, 2.8635642126552714 },
      { 5.0447993022517759, 2.1377558326431947, 3.640054944640259, 5.990826320300064, 4.6529560496527358, 4.4643028571099421, 4.2801869118065383, 0, 1.5620499351813311, 2.8017851452243807, 3.4438350715445125, 3.687817782917155, 4.651881339845203, 5.315072906367325, 6.3568860301251284, 7.0661163307718047, 4.2801869118065383, 5.724508712544683, 7.983107164506813, 6.5764732189829536, 6.0827625302982193 },
      { 4.9929950931279725, 3.4885527085024819, 5.0606323715519981, 7.5000000000000009, 5.6222771187482392, 5.3000000000000007, 4.6173585522460785, 1.5620499351813311, 0, 1.9416487838947614, 3.1400636936215172, 3.7576588456111879, 4.2426406871192865, 4.5177427992306081, 5.9908263203000631, 6.8883960397178079, 4.4944410108488446, 6.6610809933523534, 9.0271811768680035, 8.1123362849428275, 5.6639209034025191 },
      { 3.2893768406797057, 3.7735924528226423, 5.2630789467763082, 7.8243210568074222, 4.9658836071740557, 4.4944410108488473, 3.306055050963308, 2.8017851452243807, 1.9416487838947614, 0, 1.4317821063276355, 2.3, 2.3600847442411892, 2.5999999999999996, 7.8243210568074222, 8.7726848797845225, 6.4350602172784672, 8.4403791384036762, 10.75174404457249, 9.108238029388561, 3.7483329627982624 },
      { 1.878829422805594, 3.5510561809129406, 4.7927027865287037, 7.2945184899347542, 3.9204591567825315, 3.3837848631377261, 1.9646882704388495, 3.4438350715445125, 3.1400636936215172, 1.4317821063276355, 0, 0.94868329805051288, 1.2083045973594577, 2.1377558326431956, 9.12633551870629, 10.026464980241041, 7.5166481891864532, 9.154780172128655, 11.360017605620161, 9.0426765949026411, 2.6419689627245817 },
      { 1.3601470508735445, 3.2140317359976405, 4.22018956920184, 6.6309878600401628, 3.041381265149111, 2.4839484696748451, 1.0198039027185573, 3.687817782917155, 3.7576588456111879, 2.3, 0.94868329805051288, 0, 1.456021977856103, 2.7294688127912354, 9.7247107926148644, 10.568348972285122, 7.9246451024635789, 9.2590496272565677, 11.365298060323804, 8.6284413424441855, 2.6305892875931809 },
      { 1.2369316876852983, 4.6010868281309367, 5.6753854494650851, 8.0777472107017569, 4.360045871318329, 3.7802116342871606, 2.12602916254693, 4.651881339845203, 4.2426406871192865, 2.3600847442411892, 1.2083045973594577, 1.456021977856103, 0, 1.3453624047073711, 10.182828683622247, 11.119802156513398, 8.6833173384369626, 10.361949623502325, 12.557467897629682, 10.060318086422516, 1.4422205101855958 },
      { 2.4698178070456942, 5.6885850613311577, 6.89492567037528, 9.35200513259055, 5.7008771254956905, 5.12249938994628, 3.4713109915419564, 5.315072906367325, 4.5177427992306081, 2.5999999999999996, 2.1377558326431956, 2.7294688127912354, 1.3453624047073711, 0, 10.183319694480774, 11.194641575325221, 9.00499861188218, 11.016351483136328, 13.298120167903432, 11.180339887498949, 1.7117242768623691 },
      { 10.983624174196786, 8.2764726786234259, 9.4339811320566049, 11.065260954898443, 10.963576058932597, 10.815729286552989, 10.509519494249012, 6.3568860301251284, 5.9908263203000631, 7.8243210568074222, 9.12633551870629, 9.7247107926148644, 10.182828683622247, 10.183319694480774, 0, 1.1401754250991374, 2.6476404589747466, 5.2630789467763073, 7.1400280111495373, 9.8234413521942532, 11.562439189029277 },
      { 11.860016863394419, 8.85437744847146, 9.8843310345212529, 11.300442469213319, 11.576700738984314, 11.476933388322857, 11.301769772916098, 7.0661163307718047, 6.8883960397178079, 8.7726848797845225, 10.026464980241041, 10.568348972285122, 11.119802156513398, 11.194641575325221, 1.1401754250991374, 0, 2.96141857899217, 4.919349550499537, 6.5053823869162377, 9.7529482721892862, 12.52078272313676 },
      { 9.2633687176966024, 5.9236812878479537, 6.92314957226839, 8.4291162051546049, 8.6539008545279739, 8.5866174946832228, 8.5603738236130766, 4.2801869118065383, 4.4944410108488446, 6.4350602172784672, 7.5166481891864532, 7.9246451024635789, 8.6833173384369626, 9.00499861188218, 2.6476404589747466, 2.96141857899217, 0, 3.1764760348537169, 5.470831746635973, 7.2449982746719819, 10.121264743103996 },
      { 10.606601717798211, 6.4031242374328468, 6.700746227100379, 7.2124891681027838, 8.9140338792266185, 9.0553851381374155, 9.5963534741067118, 5.724508712544683, 6.6610809933523534, 8.4403791384036762, 9.154780172128655, 9.2590496272565677, 10.361949623502325, 11.016351483136328, 5.2630789467763073, 4.919349550499537, 3.1764760348537169, 0, 2.4083189157584615, 5.01597448159378, 11.778370006074693 },
      { 12.684636376341263, 8.3186537372341682, 8.2619610262939389, 8.0709355095924291, 10.580170130957253, 10.816653826391969, 11.583177456984764, 7.983107164506813, 9.0271811768680035, 10.75174404457249, 11.360017605620161, 11.365298060323804, 12.557467897629682, 13.298120167903432, 7.1400280111495373, 6.5053823869162377, 5.470831746635973, 2.4083189157584615, 0, 5.2, 13.945967159003352 },
      { 9.7005154502222215, 5.4918120870983929, 4.5099889135118723, 3.1400636936215167, 6.6708320320631671, 7.1021123618258812, 8.40535543567314, 6.5764732189829536, 8.1123362849428275, 9.108238029388561, 9.0426765949026411, 8.6284413424441855, 10.060318086422516, 11.180339887498949, 9.8234413521942532, 9.7529482721892862, 7.2449982746719819, 5.01597448159378, 5.2, 0, 11.204017136723776 },
      { 1.5524174696260025, 5.8420886675914128, 6.7268120235368558, 8.9738509013689338, 5.0328918128646487, 4.4598206241955527, 2.8635642126552714, 6.0827625302982193, 5.6639209034025191, 3.7483329627982624, 2.6419689627245817, 2.6305892875931809, 1.4422205101855958, 1.7117242768623691, 11.562439189029277, 12.52078272313676, 10.121264743103996, 11.778370006074693, 13.945967159003352, 11.204017136723776, 0 }
    };
  }
}
