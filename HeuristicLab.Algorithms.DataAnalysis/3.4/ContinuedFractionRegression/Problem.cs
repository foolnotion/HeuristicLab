using System;
using System.Collections.Generic;
using System.Linq;
using HEAL.Attic;
using HeuristicLab.Analysis;
using HeuristicLab.Common;
using HeuristicLab.Core;
using HeuristicLab.Data;
using HeuristicLab.Encodings.BinaryVectorEncoding;
using HeuristicLab.Encodings.RealVectorEncoding;
using HeuristicLab.Optimization;
using HeuristicLab.Parameters;
using HeuristicLab.Problems.DataAnalysis;
using HeuristicLab.Problems.Instances;

namespace HeuristicLab.Algorithms.DataAnalysis.ContinuedFractionRegression {
  [Item("Continued Fraction Regression (CFR)", "Attempts to find a continued fraction minimizing the regularized MSE / R^2 to given data.")]
  [StorableType("CAC0F743-8524-436C-B81E-7C628A302DF8")]
  [Creatable(CreatableAttribute.Categories.DataAnalysisRegression, Priority = 999)]
  public class Problem : HeuristicLab.Optimization.SingleObjectiveBasicProblem<MultiEncoding> 
    /*, IProblemInstanceConsumer<IRegressionProblemData>, IProblemInstanceExporter<IRegressionProblemData>*/  // only if we can change the code to work with a single dataset
    {
    private const double epsilon = 1e-6;
    enum Fitting { MSE, RegMSE, ElasticNet }
    private readonly Fitting fit = Fitting.ElasticNet;

    #region parameter properties
    public IFixedValueParameter<DoubleValue> L1L2MixingParameter {
      get { return (IFixedValueParameter<DoubleValue>)Parameters["L1L2_mixing"]; }
    }
    public IFixedValueParameter<DoubleValue> L1L2WeightParameter {
      get { return (IFixedValueParameter<DoubleValue>)Parameters["L1L2_weight"]; }
    }
    public IFixedValueParameter<DoubleValue> PenaltyParameter {
      get { return (IFixedValueParameter<DoubleValue>)Parameters["penalty"]; }
    }
    public IFixedValueParameter<IntValue> FractionDepthParameter {
      get { return (IFixedValueParameter<IntValue>)Parameters["fraction_depth"]; }
    }
    public IValueParameter<Dataset> DatasetParameter {
      get { return (IValueParameter<Dataset>)Parameters["dataset"]; }
    }
    public IValueParameter<Dataset> TrainingTrainingDatasetParameter {
      get { return (IValueParameter<Dataset>)Parameters["datasetTrainingTraining"]; }
    }
    public IValueParameter<Dataset> TrainingTestDatasetParameter {
      get { return (IValueParameter<Dataset>)Parameters["datasetTrainingTest"]; }
    }
    public IValueParameter<Dataset> Phi0_08DatasetParameter {
      get { return (IValueParameter<Dataset>)Parameters["data_phi_0_08"]; }
    }
    #endregion

    #region properties
    public double L1L2Mixing {
      get { return L1L2MixingParameter.Value.Value; }
      set { L1L2MixingParameter.Value.Value = value; }
    }
    public double L1L2Weight {
      get { return L1L2WeightParameter.Value.Value; }
      set { L1L2WeightParameter.Value.Value = value; }
    }
    public double Penalty {
      get { return PenaltyParameter.Value.Value; }
      set { PenaltyParameter.Value.Value = value; }
    }
    public int FractionDepth {
      get { return FractionDepthParameter.Value.Value; }
      set { FractionDepthParameter.Value.Value = value; }
    }
    public Dataset Dataset {
      get { return DatasetParameter.Value; }
      set { DatasetParameter.Value = value; }
    }
    public Dataset TrainingTrainingDataset {
      get { return TrainingTrainingDatasetParameter.Value; }
      set { TrainingTrainingDatasetParameter.Value = value; }
    }
    public Dataset TrainingTestDataset {
      get { return TrainingTestDatasetParameter.Value; }
      set { TrainingTestDatasetParameter.Value = value; }
    }
    public Dataset Phi0_08Dataset {
      get { return Phi0_08DatasetParameter.Value; }
      set { Phi0_08DatasetParameter.Value = value; }
    }
    #endregion


    public override bool Maximization { get { return false; } }

    public IRegressionProblemData ProblemData { get; set; }

    // BEWARE local fields
    // DO NOT store or clone
    // these variables are initialized on the first call of Evaluate. 
    // whenever a dataset parameter is changed (which has an effect on these variables)
    // all variables must be cleared to ensure that they are recalculated on the next call of Evaluate
    private DoubleMatrix dataMatrix; // data for training
    private DoubleMatrix dataNelderMead; // data for optimizing coefficients
    private DoubleMatrix dataMatrixTest; // data for testing
    private Transformation transformation;
    private double arithmeticMean;
    private double SQT; //Sum of Squares Total, short: SQT
    private void ClearLocalVariables() {
      dataMatrix = null;
      dataNelderMead = null;
      dataMatrixTest = null;
      transformation = null;
      arithmeticMean = 0.0;
      SQT = 0.0;
    }
    // END local fields


    // cloning ctor
    public Problem(Problem orig, Cloner cloner) : base(orig, cloner) {
    }

    public Problem() : base() {
      Parameters.Add(new FixedValueParameter<DoubleValue>("L1L2_mixing", "TODO Description", new DoubleValue(0.2)));
      Parameters.Add(new FixedValueParameter<DoubleValue>("L1L2_weight", "TODO Description", new DoubleValue(1)));
      Parameters.Add(new FixedValueParameter<DoubleValue>("penality", "TODO Description", new DoubleValue(0.01)));
      Parameters.Add(new FixedValueParameter<IntValue>("fraction_depth", "TODO Description", new IntValue(4)));
      Parameters.Add(new ValueParameter<Dataset>("dataset", "TODO Description", new Dataset()));
      Parameters.Add(new ValueParameter<Dataset>("datasetTrainingTraining", "TODO Description", new Dataset()));
      Parameters.Add(new ValueParameter<Dataset>("data_phi_0_08", "TODO Description", new Dataset()));
      Parameters.Add(new ValueParameter<Dataset>("datasetTrainingTest", "TODO Description", new Dataset()));

      foreach (var temperature in new[] { 350, 375, 400, 425, 450, 475, 500 }) {
        foreach (var phip in new[] { "0_001", "0_01", "0_1", "1" }) {
          // TODO: refactor
          Parameters.Add(new ValueParameter<Dataset>("data_" + temperature + "_" + phip, "TODO Description", new Dataset()));
        }
      }

      ClearLocalVariables();

      DatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      TrainingTrainingDatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      TrainingTestDatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      Phi0_08DatasetParameter.ValueChanged += DatasetParameter_ValueChanged;

      UpdateEncoding();
    }

    [StorableHook(HookType.AfterDeserialization)]
    private void AfterDeserialization() {
      DatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      TrainingTrainingDatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      TrainingTestDatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
      Phi0_08DatasetParameter.ValueChanged += DatasetParameter_ValueChanged;
    }

    #region event handlers for clearing local variables whenever a dataset is changed
    private void DatasetParameter_ValueChanged(object sender, EventArgs e) {
      ClearLocalVariables();
      UpdateEncoding();
    }

    private void UpdateEncoding() {
      int fraction_depth = FractionDepth;
      int number_variables = Dataset.Columns;
      int vector_length = (2 * fraction_depth - 1) * number_variables; //number of variables n because n-1 coefficients + 1 constant

      Encoding = new MultiEncoding()
        .Add(new RealVectorEncoding("coeff", length: vector_length, min: -3, max: 3))
        .Add(new BinaryVectorEncoding("binar", length: vector_length))
      ;
    }
    #endregion

    public override IDeepCloneable Clone(Cloner cloner) {
      return new Problem(this, cloner);
    }

    public override double Evaluate(Individual individual, IRandom random) {
      if(dataMatrix == null) {
        InitializeTransformedDatasets();
      }

      double LS = 0.0; //sum of squared residuals -> SQR = LS
      double quality = 0.0;

      double[] coeff = individual.RealVector("coeff").ToArray();
      bool[] binar = individual.BinaryVector("binar").ToArray();

      LS = calculateLS(coeff, binar, dataMatrix);
      quality = calculateQuality(coeff, binar, LS);

      #region Nelder-Mead-Methode
      double uniformPeturbation = 1.0;
      double tolerance = 1e-3;
      int maxEvals = 250;
      int numSearches = 4;
      int numPoints = dataMatrix.Rows / 5; // 20% of the training samples

      double[] optimizedCoeffShort = new double[numVariablesUsed(binar)];
      double[] optimizedCoeffLong = new double[coeff.Length];

      for (int count = 0; count < numSearches; count++) {
        int indexShort = 0;
        for (int indexLong = 0; indexLong < coeff.Length; indexLong++) {
          if (binar[indexLong] == true) {
            optimizedCoeffShort[indexShort] = coeff[indexLong];
            indexShort++;
          }
        }

        SimplexConstant[] constants = new SimplexConstant[numVariablesUsed(binar)];
        for (int i = 0; i < numVariablesUsed(binar); i++) {
          constants[i] = new SimplexConstant(optimizedCoeffShort[i], uniformPeturbation);
        }

        //ObjFunctionNelderMead NelderMead = new ObjFunctionNelderMead(this, dataNelderMead, coeff, binar);
        ObjFunctionNelderMead NelderMead = new ObjFunctionNelderMead(this, dataMatrix, numPoints, coeff, binar);

        ObjectiveFunctionDelegate objFunction = new ObjectiveFunctionDelegate(NelderMead.objFunctionNelderMead);
        RegressionResult result = NelderMeadSimplex.Regress(constants, tolerance, maxEvals, objFunction);

        optimizedCoeffShort = result.Constants;
        int shortIndex = 0;
        for (int i = 0; i < coeff.Length; i++) {
          if (binar[i] == true) {
            optimizedCoeffLong[i] = optimizedCoeffShort[shortIndex];
            shortIndex++;
          } else {
            optimizedCoeffLong[i] = coeff[i];
          }
        }

        double newLS = calculateLS(optimizedCoeffLong, binar, dataMatrix);
        double newQuality = calculateQuality(optimizedCoeffLong, binar, newLS);

        if (newQuality < quality) {
          // dont set new coeff!
          individual["coeff"] = new RealVector(optimizedCoeffLong);
          LS = newLS;
          quality = newQuality;
        }
      }
      #endregion

      //doubleThrowInvOrNaN(quality, "quality");
      individual["LS"] = new DoubleValue(LS); //Least Squares - Sum of Squares Residual, short: SQR
      individual["Depth"] = new DoubleValue(calculateDepth(coeff, binar));
      individual["CoeffNumber"] = new DoubleValue((double)numVariablesUsed(binar));
      individual["MSE"] = new DoubleValue(LS / dataMatrix.Rows);
      individual["MSETest"] = new DoubleValue(0.0); // only calculated for the best solution in a generation (see Analyze())
      individual["R2"] = new DoubleValue(1.0 - LS / SQT); //R^2 = 1 - SQR/SQT

      return quality;
    }

    private void InitializeTransformedDatasets() {
      //dataMatrix = new DoubleMatrix(this.createMatrix("dataset", 1000));
      dataMatrix = new DoubleMatrix(TrainingTrainingDataset.ToArray(TrainingTrainingDataset.VariableNames, Enumerable.Range(0, TrainingTrainingDataset.Rows)));
      dataNelderMead = new DoubleMatrix(Phi0_08Dataset.ToArray(Phi0_08Dataset.VariableNames, Enumerable.Range(0, Phi0_08Dataset.Rows)));
      dataMatrixTest = new DoubleMatrix(TrainingTestDataset.ToArray(TrainingTestDataset.VariableNames, Enumerable.Range(0, TrainingTestDataset.Rows)));
      //minMaxTransformation(0);
      //minMaxTransformation(1);
      //log10Transformation(2); // no negativ values!
      //minMaxTransformation(2); // Min and Max have to be different!

      transformation = new Transformation(dataMatrix);
      // vars["dataMatrix"] = dataMatrix;

      transformation.useSameTransformation(dataNelderMead);
      // vars["dataNelderMead"] = dataNelderMead;

      transformation.useSameTransformation(dataMatrixTest);
      // vars["dataMatrixTest"] = dataMatrixTest;

      arithmeticMean = calculateArithmeticMean();
      SQT = calculateSQT(arithmeticMean);
    }

    public override void Analyze(Individual[] individuals, double[] qualities, ResultCollection results, IRandom random) {
      // Use vars.yourVariable to access variables in the variable store i.e. yourVariable
      // Write or update results given the range of vectors and resulting qualities

      Individual bestIndividual = null;
      double bestQuality = double.MaxValue;
      int theBest = 0;

      for (int i = 0; i < qualities.Count(); i++) {
        if (qualities[i] < bestQuality) {
          bestQuality = qualities[i];
          bestIndividual = individuals[i].Copy();
          theBest = i;
        }
      }

      /*
      // Uncomment the following lines if you want to retrieve the best individual
      var orderedIndividuals = individuals.Zip(qualities, (i, q) => new { Individual = i, Quality = q }).OrderBy(z => z.Quality);
      var best = Maximization ? orderedIndividuals.Last().Individual : orderedIndividuals.First().Individual;
      if (!results.ContainsKey("Best Solution")) {
        results.Add(new Result("Best Solution", typeof(RealVector)));
        
      }
      results["Best Solution"].Value = (IItem)best.RealVector("coeff").Clone();
      //
      */

      bool[] binar = bestIndividual.BinaryVector("binar").ToArray();
      double[] coeff = bestIndividual.RealVector("coeff").ToArray();

      // calculate and set test-MSE (only) for the individual with best training-MSE
      double bestLSTest = calculateLS(coeff, binar, dataMatrixTest);
      double bestQualityTest = calculateQuality(coeff, binar, bestLSTest);
      individuals[theBest]["MSETest"] = new DoubleValue(bestQualityTest);

      InitDataTables(results);
      InitScatterPlots(results);

      #region set dataTables      
      var datatable = (DataTable)results["datatable"].Value;
      var coefficients = (DataTable)results["coefficients"].Value;
      var pearsonR2 = (DataTable)results["personR2"].Value;

      datatable.Rows["MSE"].Values.Add(calculateRegMSE(((DoubleValue)bestIndividual["LS"]).Value, binar, 0.0));
      datatable.Rows["RegMSE"].Values.Add(calculateRegMSE(((DoubleValue)bestIndividual["LS"]).Value, binar, Penalty));
      datatable.Rows["ElasticNet"].Values.Add(calculateElasticNet(((DoubleValue)bestIndividual["LS"]).Value, coeff, binar));

      coefficients.Rows["Depth"].Values.Add(((DoubleValue)bestIndividual["Depth"]).Value);
      coefficients.Rows["Number"].Values.Add(((DoubleValue)bestIndividual["CoeffNumber"]).Value);

      pearsonR2.Rows["R2"].Values.Add(((DoubleValue)bestIndividual["R2"]).Value);
      #endregion

      #region set scatterPlots
      var curves0_001 = (ScatterPlot)results["Curves0_001"].Value;
      var curves0_01 = (ScatterPlot)results["Curves0_01"].Value;
      var curves0_1 = (ScatterPlot)results["Curves0_1"].Value;
      var curves1 = (ScatterPlot)results["Curves1"].Value;
      #region curves 0.001
      curves0_001.Rows["Temp350Convergence"].Points.Clear();
      curves0_001.Rows["Temp375Convergence"].Points.Clear();
      curves0_001.Rows["Temp400Convergence"].Points.Clear();
      curves0_001.Rows["Temp425Convergence"].Points.Clear();
      curves0_001.Rows["Temp450Convergence"].Points.Clear();
      curves0_001.Rows["Temp475Convergence"].Points.Clear();
      curves0_001.Rows["Temp500Convergence"].Points.Clear();
      sampleCurve(curves0_001, "Temp350Convergence", coeff, binar, 350.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp375Convergence", coeff, binar, 375.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp400Convergence", coeff, binar, 400.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp425Convergence", coeff, binar, 425.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp450Convergence", coeff, binar, 450.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp475Convergence", coeff, binar, 475.0, 0.001, 100, 50.0);
      sampleCurve(curves0_001, "Temp500Convergence", coeff, binar, 500.0, 0.001, 100, 50.0);

      results.AddOrUpdateResult("Curves Phi' = 0.001", curves0_001);
      #endregion
      #region curves 0.01
      curves0_01.Rows["Temp350Convergence"].Points.Clear();
      curves0_01.Rows["Temp375Convergence"].Points.Clear();
      curves0_01.Rows["Temp400Convergence"].Points.Clear();
      curves0_01.Rows["Temp425Convergence"].Points.Clear();
      curves0_01.Rows["Temp450Convergence"].Points.Clear();
      curves0_01.Rows["Temp475Convergence"].Points.Clear();
      curves0_01.Rows["Temp500Convergence"].Points.Clear();
      sampleCurve(curves0_01, "Temp350Convergence", coeff, binar, 350.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp375Convergence", coeff, binar, 375.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp400Convergence", coeff, binar, 400.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp425Convergence", coeff, binar, 425.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp450Convergence", coeff, binar, 450.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp475Convergence", coeff, binar, 475.0, 0.01, 100, 60.0);
      sampleCurve(curves0_01, "Temp500Convergence", coeff, binar, 500.0, 0.01, 100, 60.0);

      results.AddOrUpdateResult("Curves Phi' = 0.01", curves0_01);
      #endregion
      #region curves 0.1
      curves0_1.Rows["Temp350Convergence"].Points.Clear();
      curves0_1.Rows["Temp375Convergence"].Points.Clear();
      curves0_1.Rows["Temp400Convergence"].Points.Clear();
      curves0_1.Rows["Temp425Convergence"].Points.Clear();
      curves0_1.Rows["Temp450Convergence"].Points.Clear();
      curves0_1.Rows["Temp475Convergence"].Points.Clear();
      curves0_1.Rows["Temp500Convergence"].Points.Clear();
      sampleCurve(curves0_1, "Temp350Convergence", coeff, binar, 350.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp375Convergence", coeff, binar, 375.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp400Convergence", coeff, binar, 400.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp425Convergence", coeff, binar, 425.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp450Convergence", coeff, binar, 450.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp475Convergence", coeff, binar, 475.0, 0.1, 100, 70.0);
      sampleCurve(curves0_1, "Temp500Convergence", coeff, binar, 500.0, 0.1, 100, 70.0);

      results.AddOrUpdateResult("Curves Phi' = 0.1", curves0_1);
      #endregion
      #region curves 1
      curves1.Rows["Temp350Convergence"].Points.Clear();
      curves1.Rows["Temp375Convergence"].Points.Clear();
      curves1.Rows["Temp400Convergence"].Points.Clear();
      curves1.Rows["Temp425Convergence"].Points.Clear();
      curves1.Rows["Temp450Convergence"].Points.Clear();
      curves1.Rows["Temp475Convergence"].Points.Clear();
      curves1.Rows["Temp500Convergence"].Points.Clear();
      sampleCurve(curves1, "Temp350Convergence", coeff, binar, 350.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp375Convergence", coeff, binar, 375.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp400Convergence", coeff, binar, 400.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp425Convergence", coeff, binar, 425.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp450Convergence", coeff, binar, 450.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp475Convergence", coeff, binar, 475.0, 1, 100, 80.0);
      sampleCurve(curves1, "Temp500Convergence", coeff, binar, 500.0, 1, 100, 80.0);

      results.AddOrUpdateResult("Curves Phi' = 1", curves1);
      #endregion
      #endregion
    }

    private void InitDataTables(ResultCollection results) {
      DataTable datatable, coefficients, pearsonR2;
      if (!results.ContainsKey("Datatable")) {
        datatable = new DataTable("Fitness");
        DataRow MSE = new DataRow("MSE");
        DataRow RegMSE = new DataRow("RegMSE");
        DataRow ElasticNet = new DataRow("ElasticNet");
        datatable.Rows.Add(MSE);
        datatable.Rows.Add(RegMSE);
        datatable.Rows.Add(ElasticNet);
        results.Add(new Result("Datatable", datatable));
      }


      if (!results.ContainsKey("Coefficients")) {
        coefficients = new DataTable("Coefficients");
        DataRow Depth = new DataRow("Depth");
        DataRow Number = new DataRow("Number");
        coefficients.Rows.Add(Depth);
        coefficients.Rows.Add(Number);
        results.Add(new Result("Coefficients", coefficients));
      }

      if (!results.ContainsKey("Pearson R2")) {
        pearsonR2 = new DataTable("PearsonR2");
        DataRow R2 = new DataRow("R2");
        pearsonR2.Rows.Add(R2);
        results.Add(new Result("Pearson R2", pearsonR2));
      }
    }

    private void InitScatterPlots(ResultCollection results) {
      #region curves 0.001
      ScatterPlot curves0_001 = new ScatterPlot("Curves0_001", "Kurven mit Phi'=0.001");
      readCurve(curves0_001, "data6082_350_0_001", "Temp350", System.Drawing.Color.Blue);
      readCurve(curves0_001, "data6082_375_0_001", "Temp375", System.Drawing.Color.Orange);
      readCurve(curves0_001, "data6082_400_0_001", "Temp400", System.Drawing.Color.Red);
      readCurve(curves0_001, "data6082_425_0_001", "Temp425", System.Drawing.Color.Green);
      readCurve(curves0_001, "data6082_450_0_001", "Temp450", System.Drawing.Color.Gray);
      readCurve(curves0_001, "data6082_475_0_001", "Temp475", System.Drawing.Color.Olive);
      readCurve(curves0_001, "data6082_500_0_001", "Temp500", System.Drawing.Color.Gold);

      var empty0_001 = new Point2D<double>[0];
      ScatterPlotDataRow Temp350Convergence0_001 = new ScatterPlotDataRow("Temp350Convergence", "Temp350Convergence", empty0_001);
      Temp350Convergence0_001.VisualProperties.Color = System.Drawing.Color.Blue;
      Temp350Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp350Convergence0_001);

      ScatterPlotDataRow Temp375Convergence0_001 = new ScatterPlotDataRow("Temp375Convergence", "Temp375Convergence", empty0_001);
      Temp375Convergence0_001.VisualProperties.Color = System.Drawing.Color.Orange;
      Temp375Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp375Convergence0_001);

      ScatterPlotDataRow Temp400Convergence0_001 = new ScatterPlotDataRow("Temp400Convergence", "Temp400Convergence", empty0_001);
      Temp400Convergence0_001.VisualProperties.Color = System.Drawing.Color.Red;
      Temp400Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp400Convergence0_001);

      ScatterPlotDataRow Temp425Convergence0_001 = new ScatterPlotDataRow("Temp425Convergence", "Temp425Convergence", empty0_001);
      Temp425Convergence0_001.VisualProperties.Color = System.Drawing.Color.Green;
      Temp425Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp425Convergence0_001);

      ScatterPlotDataRow Temp450Convergence0_001 = new ScatterPlotDataRow("Temp450Convergence", "Temp450Convergence", empty0_001);
      Temp450Convergence0_001.VisualProperties.Color = System.Drawing.Color.Gray;
      Temp450Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp450Convergence0_001);

      ScatterPlotDataRow Temp475Convergence0_001 = new ScatterPlotDataRow("Temp475Convergence", "Temp475Convergence", empty0_001);
      Temp475Convergence0_001.VisualProperties.Color = System.Drawing.Color.Olive;
      Temp475Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp475Convergence0_001);

      ScatterPlotDataRow Temp500Convergence0_001 = new ScatterPlotDataRow("Temp500Convergence", "Temp500Convergence", empty0_001);
      Temp500Convergence0_001.VisualProperties.Color = System.Drawing.Color.Gold;
      Temp500Convergence0_001.VisualProperties.PointSize = 2;
      curves0_001.Rows.Add(Temp500Convergence0_001);
      results.Add(new Result("Curves0_001", curves0_001));

      #endregion
      #region curves 0.01
      ScatterPlot curves0_01 = new ScatterPlot("Curves0_01", "Kurven mit Phi'=0.01");
      readCurve(curves0_01, "data6082_350_0_01", "Temp350", System.Drawing.Color.Blue);
      readCurve(curves0_01, "data6082_375_0_01", "Temp375", System.Drawing.Color.Orange);
      readCurve(curves0_01, "data6082_400_0_01", "Temp400", System.Drawing.Color.Red);
      readCurve(curves0_01, "data6082_425_0_01", "Temp425", System.Drawing.Color.Green);
      readCurve(curves0_01, "data6082_450_0_01", "Temp450", System.Drawing.Color.Gray);
      readCurve(curves0_01, "data6082_475_0_01", "Temp475", System.Drawing.Color.Olive);
      readCurve(curves0_01, "data6082_500_0_01", "Temp500", System.Drawing.Color.Gold);

      var empty0_01 = new Point2D<double>[0];
      ScatterPlotDataRow Temp350Convergence0_01 = new ScatterPlotDataRow("Temp350Convergence", "Temp350Convergence", empty0_01);
      Temp350Convergence0_01.VisualProperties.Color = System.Drawing.Color.Blue;
      Temp350Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp350Convergence0_01);

      ScatterPlotDataRow Temp375Convergence0_01 = new ScatterPlotDataRow("Temp375Convergence", "Temp375Convergence", empty0_01);
      Temp375Convergence0_01.VisualProperties.Color = System.Drawing.Color.Orange;
      Temp375Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp375Convergence0_01);

      ScatterPlotDataRow Temp400Convergence0_01 = new ScatterPlotDataRow("Temp400Convergence", "Temp400Convergence", empty0_01);
      Temp400Convergence0_01.VisualProperties.Color = System.Drawing.Color.Red;
      Temp400Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp400Convergence0_01);

      ScatterPlotDataRow Temp425Convergence0_01 = new ScatterPlotDataRow("Temp425Convergence", "Temp425Convergence", empty0_01);
      Temp425Convergence0_01.VisualProperties.Color = System.Drawing.Color.Green;
      Temp425Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp425Convergence0_01);

      ScatterPlotDataRow Temp450Convergence0_01 = new ScatterPlotDataRow("Temp450Convergence", "Temp450Convergence", empty0_01);
      Temp450Convergence0_01.VisualProperties.Color = System.Drawing.Color.Gray;
      Temp450Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp450Convergence0_01);

      ScatterPlotDataRow Temp475Convergence0_01 = new ScatterPlotDataRow("Temp475Convergence", "Temp475Convergence", empty0_01);
      Temp475Convergence0_01.VisualProperties.Color = System.Drawing.Color.Olive;
      Temp475Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp475Convergence0_01);

      ScatterPlotDataRow Temp500Convergence0_01 = new ScatterPlotDataRow("Temp500Convergence", "Temp500Convergence", empty0_01);
      Temp500Convergence0_01.VisualProperties.Color = System.Drawing.Color.Gold;
      Temp500Convergence0_01.VisualProperties.PointSize = 2;
      curves0_01.Rows.Add(Temp500Convergence0_01);
      results.Add(new Result("Curves0_01", curves0_01));

      #endregion
      #region curves 0.1
      ScatterPlot curves0_1 = new ScatterPlot("Curves0_1", "Kurven mit Phi'=0.1");
      readCurve(curves0_1, "data6082_350_0_1", "Temp350", System.Drawing.Color.Blue);
      readCurve(curves0_1, "data6082_375_0_1", "Temp375", System.Drawing.Color.Orange);
      readCurve(curves0_1, "data6082_400_0_1", "Temp400", System.Drawing.Color.Red);
      readCurve(curves0_1, "data6082_425_0_1", "Temp425", System.Drawing.Color.Green);
      readCurve(curves0_1, "data6082_450_0_1", "Temp450", System.Drawing.Color.Gray);
      readCurve(curves0_1, "data6082_475_0_1", "Temp475", System.Drawing.Color.Olive);
      readCurve(curves0_1, "data6082_500_0_1", "Temp500", System.Drawing.Color.Gold);

      var empty0_1 = new Point2D<double>[0];
      ScatterPlotDataRow Temp350Convergence0_1 = new ScatterPlotDataRow("Temp350Convergence", "Temp350Convergence", empty0_1);
      Temp350Convergence0_1.VisualProperties.Color = System.Drawing.Color.Blue;
      Temp350Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp350Convergence0_1);

      ScatterPlotDataRow Temp375Convergence0_1 = new ScatterPlotDataRow("Temp375Convergence", "Temp375Convergence", empty0_1);
      Temp375Convergence0_1.VisualProperties.Color = System.Drawing.Color.Orange;
      Temp375Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp375Convergence0_1);

      ScatterPlotDataRow Temp400Convergence0_1 = new ScatterPlotDataRow("Temp400Convergence", "Temp400Convergence", empty0_1);
      Temp400Convergence0_1.VisualProperties.Color = System.Drawing.Color.Red;
      Temp400Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp400Convergence0_1);

      ScatterPlotDataRow Temp425Convergence0_1 = new ScatterPlotDataRow("Temp425Convergence", "Temp425Convergence", empty0_1);
      Temp425Convergence0_1.VisualProperties.Color = System.Drawing.Color.Green;
      Temp425Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp425Convergence0_1);

      ScatterPlotDataRow Temp450Convergence0_1 = new ScatterPlotDataRow("Temp450Convergence", "Temp450Convergence", empty0_1);
      Temp450Convergence0_1.VisualProperties.Color = System.Drawing.Color.Gray;
      Temp450Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp450Convergence0_1);

      ScatterPlotDataRow Temp475Convergence0_1 = new ScatterPlotDataRow("Temp475Convergence", "Temp475Convergence", empty0_1);
      Temp475Convergence0_1.VisualProperties.Color = System.Drawing.Color.Olive;
      Temp475Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp475Convergence0_1);

      ScatterPlotDataRow Temp500Convergence0_1 = new ScatterPlotDataRow("Temp500Convergence", "Temp500Convergence", empty0_1);
      Temp500Convergence0_1.VisualProperties.Color = System.Drawing.Color.Gold;
      Temp500Convergence0_1.VisualProperties.PointSize = 2;
      curves0_1.Rows.Add(Temp500Convergence0_1);
      results.Add(new Result("Curves0_1", curves0_1));
      #endregion
      #region curves 1
      ScatterPlot curves1 = new ScatterPlot("Curves1", "Kurven mit Phi'=1");
      readCurve(curves1, "data6082_350_1", "Temp350", System.Drawing.Color.Blue);
      readCurve(curves1, "data6082_375_1", "Temp375", System.Drawing.Color.Orange);
      readCurve(curves1, "data6082_400_1", "Temp400", System.Drawing.Color.Red);
      readCurve(curves1, "data6082_425_1", "Temp425", System.Drawing.Color.Green);
      readCurve(curves1, "data6082_450_1", "Temp450", System.Drawing.Color.Gray);
      readCurve(curves1, "data6082_475_1", "Temp475", System.Drawing.Color.Olive);
      readCurve(curves1, "data6082_500_1", "Temp500", System.Drawing.Color.Gold);

      var empty1 = new Point2D<double>[0];
      ScatterPlotDataRow Temp350Convergence1 = new ScatterPlotDataRow("Temp350Convergence", "Temp350Convergence", empty1);
      Temp350Convergence1.VisualProperties.Color = System.Drawing.Color.Blue;
      Temp350Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp350Convergence1);

      ScatterPlotDataRow Temp375Convergence1 = new ScatterPlotDataRow("Temp375Convergence", "Temp375Convergence", empty1);
      Temp375Convergence1.VisualProperties.Color = System.Drawing.Color.Orange;
      Temp375Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp375Convergence1);

      ScatterPlotDataRow Temp400Convergence1 = new ScatterPlotDataRow("Temp400Convergence", "Temp400Convergence", empty1);
      Temp400Convergence1.VisualProperties.Color = System.Drawing.Color.Red;
      Temp400Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp400Convergence1);

      ScatterPlotDataRow Temp425Convergence1 = new ScatterPlotDataRow("Temp425Convergence", "Temp425Convergence", empty1);
      Temp425Convergence1.VisualProperties.Color = System.Drawing.Color.Green;
      Temp425Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp425Convergence1);

      ScatterPlotDataRow Temp450Convergence1 = new ScatterPlotDataRow("Temp450Convergence", "Temp450Convergence", empty1);
      Temp450Convergence1.VisualProperties.Color = System.Drawing.Color.Gray;
      Temp450Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp450Convergence1);

      ScatterPlotDataRow Temp475Convergence1 = new ScatterPlotDataRow("Temp475Convergence", "Temp475Convergence", empty1);
      Temp475Convergence1.VisualProperties.Color = System.Drawing.Color.Olive;
      Temp475Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp475Convergence1);

      ScatterPlotDataRow Temp500Convergence1 = new ScatterPlotDataRow("Temp500Convergence", "Temp500Convergence", empty1);
      Temp500Convergence1.VisualProperties.Color = System.Drawing.Color.Gold;
      Temp500Convergence1.VisualProperties.PointSize = 2;
      curves1.Rows.Add(Temp500Convergence1);
      results.Add(new Result("Curves1", curves1));
      #endregion
    }

    public override IEnumerable<Individual> GetNeighbors(Individual individual, IRandom random) {
      // Use vars.yourVariable to access variables in the variable store i.e. yourVariable
      // Create new vectors, based on the given one that represent small changes
      // This method is only called from move-based algorithms (Local Search, Simulated Annealing, etc.)
      while (true) {
        // Algorithm will draw only a finite amount of samples
        // Change to a for-loop to return a concrete amount of neighbors
        var neighbor = individual.Copy();
        // For instance, perform a single bit-flip in a binary parameter
        //var bIndex = random.Next(neighbor.BinaryVector("b").Length);
        //neighbor.BinaryVector("b")[bIndex] = !neighbor.BinaryVector("b")[bIndex];
        yield return neighbor;
      }
    }
    #region Import & Export
    public void Load(IRegressionProblemData data) {
      ProblemData = data;
    }

    public IRegressionProblemData Export() {
      return ProblemData;
    }
    #endregion

    // Implement further classes and methods

    private int numVariablesUsed(bool[] binar) {
      int num = 0;
      for (int i = 0; i < binar.Count(); i++) {
        if (binar[i] == true) // eventually we need coeff < valuenearzero too
          num++;
      }
      return num;
    }


    /* old code
    private double[,] createMatrix(string name, int numRows) {
      var dataset = (HeuristicLab.Problems.DataAnalysis.Dataset)vars[name];

      IEnumerable<string> variables = dataset.VariableNames;
      // var rows = dataset.Rows;
      var rows = Enumerable.Range(0, numRows).ToArray();

      double[,] dataMatrix = dataset.ToArray(variables, rows);

      return dataMatrix;
    }
    */


    private double evaluateContinuedFraction(double[] coeff, bool[] binar, double[] dataPoint) {
      double value = 0.0;
      bool firstCoeffUnequalZero = false;

      for (var i = coeff.Count() - 1; i > 0; i = i - 2 * dataPoint.Count()) {

        if ((linearFunctionEqualZero(coeff, binar, i - dataPoint.Count() + 1, i) == false)) {
          firstCoeffUnequalZero = true;
          value = this.evaluateLinearFunction(coeff, binar, i - dataPoint.Count() + 1, i, dataPoint) + value;
        }

        if (firstCoeffUnequalZero == true && i > 2 * dataPoint.Count()) { // don't take the first coeffVecPart and don't take the last coeffVecPart (both are only to add)
          if (linearFunctionEqualZero(coeff, binar, i - 2 * dataPoint.Count() + 1, i - dataPoint.Count()) == false) {
            if (valueNearZero(value) == true)  // no division by zero
              return double.MaxValue;
            else
              value = this.evaluateLinearFunction(coeff, binar, i - 2 * dataPoint.Count() + 1, i - dataPoint.Count(), dataPoint) / value;
          } else return double.MaxValue; // don't allow coeffVecParts in middle to be equal zero
        }
      }

      //doubleThrowInvOrNaN(value, "evaluateContinuedFractionValue");
      return value;
    }

    private bool linearFunctionEqualZero(double[] coeff, bool[] binar, int start, int end) {
      for (int i = start; i <= end; i++) {
        if (binar[i] == true && valueNearZero(coeff[i]) == false)
          return false;
      }
      return true;
    }

    private double evaluateLinearFunction(double[] coeff, bool[] binar, int start, int end, double[] dataPoint) {
      double value = 0.0;

      for (int i = start; i < end; i++) {
        if (binar[i] == true)
          value += dataPoint[i - start] * coeff[i];
      }
      if (binar[end] == true)
        value += coeff[end];

      //doubleThrowInvOrNaN(value, "evaluateLinearFunctionValue");
      return value;
    }

    private bool valueNearZero(double value) {
      if (Math.Abs(value) < epsilon)
        return true;
      else
        return false;
    }

    private double calculateDepth(double[] coeff, bool[] binar) {
      int coeffUnequalZero = -1;
      double depth = 0.0;

      for (int i = 0; i < coeff.Count(); i++) {
        if (valueNearZero(coeff[i]) == false && binar[i] == true)
          coeffUnequalZero = i;
      }
      if (coeffUnequalZero == -1) return 0.0; // equal zero
      if (coeffUnequalZero < dataMatrix.Columns) return 1.0; // linear function
      depth = (double)(coeffUnequalZero - dataMatrix.Columns + 1);
      depth = Math.Ceiling(depth / (2 * dataMatrix.Columns)) + 1.0;
      return depth;
    }

    private double calculateRegMSE(double quality, bool[] binar, double penalty) {
      return (quality / dataMatrix.Rows) * (1 + penalty * (double)numVariablesUsed(binar));
    }

    private double calculateElasticNet(double quality, double[] coeff, bool[] binar) {
      double valueL1 = 0.0;
      double valueL2 = 0.0;
      double elasticNet = 0.0;
      double L1L2weight = L1L2Weight;
      double L1L2mixing = L1L2Mixing;

      for (int i = 0; i < coeff.Count(); i++) {
        if (binar[i] == true) {
          valueL1 += Math.Abs(coeff[i]);
          valueL2 += Math.Pow(coeff[i], 2);
        }
      }
      elasticNet = quality + (L1L2weight * L1L2mixing * valueL1) + (L1L2weight * (1 - L1L2mixing) * valueL2);
      return elasticNet / dataMatrix.Rows;
    }

    public double calculateLS(double[] coeff, bool[] binar, DoubleMatrix dataMatrix) {
      double LS = 0.0;
      double[] dataPoint = new double[dataMatrix.Columns];
      for (var i = 0; i < dataMatrix.Rows; i++) {
        for (var j = 0; j < dataMatrix.Columns; j++) {
          dataPoint[j] = dataMatrix[i, j];
        }
        double continuedFractionValue = evaluateContinuedFraction(coeff, binar, dataPoint);
        if (continuedFractionValue == double.MaxValue)
          return double.MaxValue;
        else
          LS += Math.Pow((continuedFractionValue - dataMatrix[i, dataMatrix.Columns - 1]), 2);
      }
      return LS;
    }

    public double calculateQuality(double[] coeff, bool[] binar, double LS) {
      switch (fit) {
        case Fitting.MSE:
          return calculateRegMSE(LS, binar, 0.0); //MSE
        case Fitting.RegMSE:
          return calculateRegMSE(LS, binar, Penalty); //RegMSE
        case Fitting.ElasticNet:
          return calculateElasticNet(LS, coeff, binar); // Elastic Net
        default:
          return calculateElasticNet(LS, coeff, binar);
      }
    }

    private double calculateArithmeticMean() {
      double arithmeticMean = 0.0;
      for (int i = 0; i < dataMatrix.Rows; i++) {
        arithmeticMean = arithmeticMean + dataMatrix[i, dataMatrix.Columns - 1];
      }
      return arithmeticMean / dataMatrix.Rows;
    }

    private double calculateSQT(double arithmeticMean) {
      double SQT = 0.0;
      for (int i = 0; i < dataMatrix.Rows; i++) {
        SQT += Math.Pow((dataMatrix[i, dataMatrix.Columns - 1] - arithmeticMean), 2);
      }
      return SQT;
    }

    private static void doubleThrowInvOrNaN(double value, string valueName) {
      if (double.IsInfinity(value) || double.IsNaN(value))
        throw new InvalidProgramException(valueName + " is Infinity or NaN");
    }


    private void readCurve(ScatterPlot scatterPlot, string matrixName, string RowName, System.Drawing.Color color) {
      var ds = ((IValueParameter<Dataset>)Parameters[matrixName]).Value;
      DoubleMatrix dataMatrix = new DoubleMatrix(ds.ToArray(ds.VariableNames, Enumerable.Range(0, ds.Rows))); // TODO: performance / refactoring
      var points = new Point2D<double>[dataMatrix.Rows];
      for (int i = 0; i < dataMatrix.Rows; i++) {
        points[i] = new Point2D<double>(dataMatrix[i, 1], dataMatrix[i, 3]);
      }
      ScatterPlotDataRow Curve = new ScatterPlotDataRow(RowName, RowName, points);
      Curve.VisualProperties.Color = color;
      scatterPlot.Rows.Add(Curve);
    }

    private void sampleCurve(ScatterPlot scatterPlot, string scatterPlotDataRow, double[] coeff, bool[] binar, double temp, double phip, int subdivisions, double maxStress) {
      var points = new Point2D<double>[subdivisions + 1];
      double phiMin = 0.0;
      double phiMax = 0.70;
      double step = (phiMax - phiMin) / subdivisions; // subdivisions > 0 !
      double minStress = 0.0;

      double[] dataPoint = new double[3] { temp, 0.0, phip };
      //vars["sample0"] = new DoubleArray(dataPoint);
      double[] dataPointNew = new double[4];
      double x = 0.0;
      //scatterPlot.Rows[scatterPlotDataRow].Points.Clear();
      for (int i = 0; i <= subdivisions; i++) {
        x = phiMin + (double)i * step;
        dataPoint[0] = temp;
        dataPoint[1] = x;
        dataPoint[2] = phip;
        //vars["sample1"] = new DoubleArray(dataPoint);
        dataPoint = transformation.transform0(dataPoint);
        //vars["sample2"] = new DoubleArray(dataPoint);
        dataPointNew[0] = dataPoint[0];
        dataPointNew[1] = dataPoint[1];
        dataPointNew[2] = dataPoint[2];
        dataPointNew[3] = 0.0;
        points[i] = new Point2D<double>(x, evaluateContinuedFraction(coeff, binar, dataPointNew));
        if (points[i].Y >= minStress && points[i].Y <= maxStress) {
          scatterPlot.Rows[scatterPlotDataRow].Points.Add(points[i]);
        }
      }
      //scatterPlot.Rows[ScatterPlotDataRow].Point2D.Add(points);
    }
  }
}

