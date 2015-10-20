package mind

import "github.com/gonum/matrix/mat64"

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
	_, inCols := input.Dims()
	_, outCols := output.Dims()

	// Setup the weights
	m.Weights.InputHidden = Normals(inCols, m.HiddenUnits)
	m.Weights.HiddenOutput = Normals(m.HiddenUnits, outCols)

	for i := 0; i < m.Iterations; i++ {
		m.Forward(input)
		m.Back(input, output)
	}
}

// Forward propagate the examples through the network.
func (m *Mind) Forward(input *mat64.Dense) {
	HiddenSum := &mat64.Dense{}
	OutputSum := &mat64.Dense{}

	HiddenSum.Mul(input, m.Weights.InputHidden)
	m.Results.HiddenResult = m.Activate(HiddenSum)
	OutputSum.Mul(m.Results.HiddenResult, m.Weights.HiddenOutput)
	m.Results.OutputResult = m.Activate(OutputSum)

	m.Results.HiddenSum = HiddenSum
	m.Results.OutputSum = OutputSum
}

// Back propagate the error and update the weights.
func (m *Mind) Back(input *mat64.Dense, output *mat64.Dense) {
	ErrorOutputLayer := &mat64.Dense{}
	DeltaOutputLayer := &mat64.Dense{}
	HiddenOutputChanges := &mat64.Dense{}
	DeltaHiddenLayer := &mat64.Dense{}
	InputHiddenChanges := &mat64.Dense{}

	ErrorOutputLayer.Sub(output, m.Results.OutputResult)
	DeltaOutputLayer.MulElem(m.ActivatePrime(m.Results.OutputSum), ErrorOutputLayer)
	HiddenOutputChanges.Mul(m.Results.HiddenResult.T(), DeltaOutputLayer)
	HiddenOutputChanges.Scale(m.LearningRate, HiddenOutputChanges)
	m.Weights.HiddenOutput.Add(HiddenOutputChanges, m.Weights.HiddenOutput)

	DeltaHiddenLayer.Mul(DeltaOutputLayer, m.Weights.HiddenOutput.T())
	DeltaHiddenLayer.MulElem(m.ActivatePrime(m.Results.HiddenSum), DeltaHiddenLayer)
	InputHiddenChanges.Mul(input.T(), DeltaHiddenLayer)
	InputHiddenChanges.Scale(m.LearningRate, InputHiddenChanges)
	m.Weights.InputHidden.Add(InputHiddenChanges, m.Weights.InputHidden)
}

// Predict from input.
func (m *Mind) Predict(input [][]float64) *mat64.Dense {
	var in []float64

	for _, i := range input {
		in = append(in, i...)
	}

	m.Forward(mat64.NewDense(len(input), len(input[0]), in))
	return m.Results.OutputResult
}
