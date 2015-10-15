package mind

import (
	"github.com/gonum/matrix/mat64"
)

// Version.
const Version = "0.0.1"

// Mind represents the neural network.
type Mind struct {
	LearningRate  float64                         // speed the network will learn at
	Iterations    int                             // number of training iterations
	HiddenUnits   int                             // number of units in hidden layer
	Activate      func(*mat64.Dense) *mat64.Dense // activation function
	ActivatePrime func(*mat64.Dense) *mat64.Dense // derivative of activation function
	Weights                                       // learning weights
	Results                                       // learning results
}

// Weights represents the connections between units.
type Weights struct {
	InputHidden  *mat64.Dense
	HiddenOutput *mat64.Dense
}

// Results represents, at a given unit, the output of multiplying
// the inputs and weights in all previous layers.
type Results struct {
	HiddenSum    *mat64.Dense
	HiddenResult *mat64.Dense
	OutputSum    *mat64.Dense
	OutputResult *mat64.Dense
}

// New mind loaded with `rate`, `iterations`, and `units`.
func New(rate float64, iterations int, units int, activator string) *Mind {
	m := &Mind{
		LearningRate: rate,
		Iterations:   iterations,
		HiddenUnits:  units,
	}

	switch activator {
	case "sigmoid":
		m.Activate = Activator(Sigmoid)
		m.ActivatePrime = Activator(SigmoidPrime)
	case "htan":
		m.Activate = Activator(Htan)
		m.ActivatePrime = Activator(Htanprime)
	default:
		panic("unknown activator " + activator)
	}

	return m
}

// Learn from examples.
func (m *Mind) Learn(examples [][][]float64) {
	input, output := Format(examples)

	// Setup the weights
	_, inputcols := input.RowView(0).Dims()
	m.Weights.InputHidden = Normals(inputcols, m.HiddenUnits)
	_, outputcols := output.Dims()
	m.Weights.HiddenOutput = Normals(m.HiddenUnits, outputcols)

	for i := 0; i < m.Iterations; i++ {
		m.Forward(input)
		m.Back(input, output)
	}
}

// Forward propagate the examples through the network.
func (m *Mind) Forward(input *mat64.Dense) {

	m.Results.HiddenSum.Product(input, m.Weights.InputHidden)
	m.Results.HiddenResult = m.Activate(m.Results.HiddenSum)
	m.Results.OutputSum = mat64.NewDense(1, 1, nil)
	m.Results.OutputSum.Product(m.Results.HiddenResult, m.Weights.HiddenOutput)
	m.Results.OutputResult = m.Activate(m.Results.OutputSum)
}

// Back propagate the error and update the weights.
func (m *Mind) Back(input *mat64.Dense, output *mat64.Dense) {
	ErrorOutputLayer := mat64.NewDense(1, 1, nil)
	ErrorOutputLayer.Sub(output, m.Results.OutputResult)
	DeltaOutputLayer := m.ActivatePrime(m.Results.OutputSum)
	DeltaOutputLayer.MulElem(DeltaOutputLayer, ErrorOutputLayer)
	HiddenOutputChanges := mat64.NewDense(1, 1, nil)
	HiddenOutputChanges.Product(m.Results.HiddenResult.T(), DeltaOutputLayer)
	HiddenOutputChanges.Scale(m.LearningRate, HiddenOutputChanges)
	m.Weights.HiddenOutput.Add(m.Weights.HiddenOutput, HiddenOutputChanges)
	DeltaHiddenLayer := mat64.NewDense(1, 1, nil)
	DeltaHiddenLayer.Product(DeltaHiddenLayer, DeltaOutputLayer, m.Weights.HiddenOutput.T())
	DeltaHiddenLayer.MulElem(DeltaHiddenLayer, m.ActivatePrime(m.Results.HiddenSum))
	InputHiddenChanges := mat64.NewDense(1, 1, nil)
	InputHiddenChanges.Product(InputHiddenChanges, input.T(), DeltaHiddenLayer)
	InputHiddenChanges.Scale(m.LearningRate, InputHiddenChanges)
	m.Weights.InputHidden.Add(m.Weights.InputHidden, InputHiddenChanges)
}

// Predict from input.
func (m *Mind) Predict(input [][]float64) *mat64.Dense {
	m.Forward(MakeDenseMatrixStacked(input))
	return m.Results.OutputResult
}
