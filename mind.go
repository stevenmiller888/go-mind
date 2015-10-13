package mind

import "github.com/skelterjohn/go.matrix"

// Version.
const Version = "0.0.1"

// Mind represents the neural network.
type Mind struct {
	LearningRate  float64                                       // speed the network will learn at
	Iterations    int                                           // number of training iterations
	HiddenUnits   int                                           // number of units in hidden layer
	Activate      func(*matrix.DenseMatrix) *matrix.DenseMatrix // activation function
	ActivatePrime func(*matrix.DenseMatrix) *matrix.DenseMatrix // derivative of activation function
	Weights                                                     // learning weights
	Results                                                     // learning results
}

// Weights represents the connections between units.
type Weights struct {
	InputHidden  *matrix.DenseMatrix
	HiddenOutput *matrix.DenseMatrix
}

// Results represents, at a given unit, the output of multiplying
// the inputs and weights in all previous layers.
type Results struct {
	HiddenSum    *matrix.DenseMatrix
	HiddenResult *matrix.DenseMatrix
	OutputSum    *matrix.DenseMatrix
	OutputResult *matrix.DenseMatrix
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
	m.Weights.InputHidden = Normals(input.GetRowVector(0).Cols(), m.HiddenUnits)
	m.Weights.HiddenOutput = Normals(m.HiddenUnits, output.Cols())

	for i := 0; i < m.Iterations; i++ {
		m.Forward(input)
		m.Back(input, output)
	}
}

// Forward propagate the examples through the network.
func (m *Mind) Forward(input *matrix.DenseMatrix) {
	m.Results.HiddenSum = matrix.Product(input, m.Weights.InputHidden)
	m.Results.HiddenResult = m.Activate(m.Results.HiddenSum)
	m.Results.OutputSum = matrix.Product(m.Results.HiddenResult, m.Weights.HiddenOutput)
	m.Results.OutputResult = m.Activate(m.Results.OutputSum)
}

// Back propagate the error and update the weights.
func (m *Mind) Back(input *matrix.DenseMatrix, output *matrix.DenseMatrix) {
	ErrorOutputLayer := matrix.Difference(output, m.Results.OutputResult)

	DeltaOutputLayer, _ := m.ActivatePrime(m.Results.OutputSum).ElementMult(ErrorOutputLayer)
	HiddenOutputChanges := matrix.Product(m.Results.HiddenResult.Transpose(), DeltaOutputLayer)
	HiddenOutputChanges.Scale(m.LearningRate)
	m.Weights.HiddenOutput.Add(HiddenOutputChanges)

	DeltaHiddenLayer, _ := matrix.Product(DeltaOutputLayer, m.Weights.HiddenOutput.Transpose()).ElementMult(m.ActivatePrime(m.Results.HiddenSum))
	InputHiddenChanges := matrix.Product(input.Transpose(), DeltaHiddenLayer)
	InputHiddenChanges.Scale(m.LearningRate)
	m.Weights.InputHidden.Add(InputHiddenChanges)
}

// Predict from input.
func (m *Mind) Predict(input [][]float64) *matrix.DenseMatrix {
	m.Forward(matrix.MakeDenseMatrixStacked(input))
	return m.Results.OutputResult
}
