package mind

import (
	"github.com/gonum/matrix/mat64"
	"log"
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
	//This library uses Column-vectors, while rows are used for use in the Normals() function
	//as a result, these two are switched.
	inputcols, _ := input.RowView(0).Dims()
	m.Weights.InputHidden = Normals(inputcols, m.HiddenUnits)
	_, outputcols := output.Dims()
	//log.Println("Output dims:", or, outputcols)
	m.Weights.HiddenOutput = Normals(m.HiddenUnits, outputcols)
	log.Println("outputcols: ", outputcols, "; HiddenUnits: ", m.HiddenUnits)
	for i := 0; i < m.Iterations; i++ {
		m.Forward(input)
		m.Back(input, output)
	}
}

// Forward propagate the examples through the network.
func (m *Mind) Forward(in *mat64.Dense) {
	input := mat64.DenseCopyOf(in)
	m.Results.HiddenSum = mat64.NewDense(1, 1, nil)

	ir, ic := input.Dims()
	or, oc := m.Weights.InputHidden.Dims()
	log.Println("input dims(r,c):", ir, ic)
	log.Println("InputHidden dims(r,c):", or, oc)

	input.Product(m.Weights.InputHidden)
	m.Results.HiddenSum = mat64.DenseCopyOf(input)
	m.Results.HiddenResult = m.Activate(m.Results.HiddenSum)
	//m.Results.OutputSum = mat64.NewDense(1, 1, nil)
	m.Results.HiddenResult.Product(m.Weights.HiddenOutput)
	m.Results.OutputSum = mat64.DenseCopyOf(m.Results.HiddenResult)
	m.Results.OutputResult = m.Activate(m.Results.OutputSum)
}

// Back propagate the error and update the weights.
func (m *Mind) Back(input *mat64.Dense, output *mat64.Dense) {
	ErrorOutputLayer := mat64.NewDense(1, 1, nil)
	ErrorOutputLayer.Sub(output, m.Results.OutputResult)
	DeltaOutputLayer := m.ActivatePrime(m.Results.OutputSum)
	DeltaOutputLayer.MulElem(DeltaOutputLayer, ErrorOutputLayer)

	HiddenOutputChanges := mat64.DenseCopyOf(m.Results.HiddenResult.T())
	HiddenOutputChanges.Product(DeltaOutputLayer)
	HiddenOutputChanges.Scale(m.LearningRate, HiddenOutputChanges)
	m.Weights.HiddenOutput.Add(m.Weights.HiddenOutput, HiddenOutputChanges)

	DeltaHiddenLayer := mat64.DenseCopyOf(DeltaOutputLayer)
	DeltaHiddenLayer.Product(DeltaOutputLayer, m.Weights.HiddenOutput.T())
	DeltaHiddenLayer.MulElem(DeltaHiddenLayer, m.ActivatePrime(m.Results.HiddenSum))

	InputHiddenChanges := mat64.DenseCopyOf(input.T())
	InputHiddenChanges.Product(DeltaHiddenLayer)
	InputHiddenChanges.Scale(m.LearningRate, InputHiddenChanges)
	m.Weights.InputHidden.Add(m.Weights.InputHidden, InputHiddenChanges)
}

// Predict from input.
func (m *Mind) Predict(input [][]float64) *mat64.Dense {
	m.Forward(MakeDenseMatrixStacked(input))
	return m.Results.OutputResult
}
