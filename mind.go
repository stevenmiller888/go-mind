package mind

import (
	"math"
	"math/rand"
	"time"

	"github.com/skelterjohn/go.matrix"
)

// Version
const Version = "0.0.1"

// Mind struct
type Mind struct {
	Activator    string                         // activation function, `sigmoid` or `htan`
	LearningRate float64                        // speed the network will learn at
	Iterations   int                            // number of training iterations
	HiddenUnits  int                            // number of units in hidden layer
	Weights      map[string]*matrix.DenseMatrix // learning weights
	Results      map[string]*matrix.DenseMatrix // learning results
}

// New Mind
func New(activator string, rate float64, iterations int, units int) *Mind {
	return &Mind{
		Activator:    activator,
		LearningRate: rate,
		Iterations:   iterations,
		HiddenUnits:  units,
		Weights:      map[string]*matrix.DenseMatrix{},
		Results:      map[string]*matrix.DenseMatrix{},
	}
}

// Learn from examples
func (m *Mind) Learn(examples [][][]float64) {
	normalized := Normalize(examples)

	// Setup the weights
	m.Weights["InputHidden"] = Normals(normalized["Input"].GetRowVector(0).Cols(), m.HiddenUnits)
	m.Weights["HiddenOutput"] = Normals(m.HiddenUnits, normalized["Output"].Cols())

	for i := 0; i < m.Iterations; i++ {
		m.Forward(normalized)
		m.Back(normalized)
	}
}

// Forward propagate
func (m *Mind) Forward(examples map[string]*matrix.DenseMatrix) {
	m.Results["HiddenSum"] = matrix.Product(examples["Input"], m.Weights["InputHidden"])
	m.Results["HiddenResult"] = MatrixSigmoid(m.Results["HiddenSum"])
	m.Results["OutputSum"] = matrix.Product(m.Results["HiddenResult"], m.Weights["HiddenOutput"])
	m.Results["OutputResult"] = MatrixSigmoid(m.Results["OutputSum"])
}

// Back propagate
func (m *Mind) Back(examples map[string]*matrix.DenseMatrix) {
	ErrorOutputLayer := matrix.Difference(examples["Output"], m.Results["OutputResult"])

	MatrixSigmoidPrime(m.Results["OutputSum"])
	DeltaOutputLayer, _ := m.Results["OutputSum"].ElementMult(ErrorOutputLayer)

	HiddenOutputChanges := matrix.Product(m.Results["HiddenResult"].Transpose(), DeltaOutputLayer)
	HiddenOutputChanges.Scale(m.LearningRate)

	MatrixSigmoidPrime(m.Results["HiddenSum"])
	DeltaHiddenLayer, _ := matrix.Product(DeltaOutputLayer, m.Weights["HiddenOutput"].Transpose()).ElementMult(m.Results["HiddenSum"])

	InputHiddenChanges := matrix.Product(examples["Input"].Transpose(), DeltaHiddenLayer)
	InputHiddenChanges.Scale(m.LearningRate)

	m.Weights["InputHidden"].Add(InputHiddenChanges)
	m.Weights["HiddenOutput"].Add(HiddenOutputChanges)
}

// Predict from input
func (m *Mind) Predict(input [][]float64) *matrix.DenseMatrix {
	m.Forward(map[string]*matrix.DenseMatrix{"Input": matrix.MakeDenseMatrixStacked(input)})
	return m.Results["OutputResult"]
}

// Normalize the examples
func Normalize(examples [][][]float64) map[string]*matrix.DenseMatrix {
	var output, input [][]float64
	var ret = map[string]*matrix.DenseMatrix{}

	for _, example := range examples {
		output = append(output, example[1])
		input = append(input, example[0])
	}

	ret["Output"] = matrix.MakeDenseMatrixStacked(output)
	ret["Input"] = matrix.MakeDenseMatrixStacked(input)

	return ret
}

// Normals returns a DenseMatrix filled with random values
func Normals(rows, cols int) *matrix.DenseMatrix {
	ret := matrix.Zeros(rows, cols)

	for i := 0; i < ret.Rows(); i++ {
		for j := 0; j < ret.Cols(); j++ {
			rand.Seed(time.Now().UTC().UnixNano())
			ret.Set(i, j, rand.NormFloat64())
		}
	}

	return ret
}

// MatrixSigmoid calculates the sigmoid of each element in `m1`
func MatrixSigmoid(m1 *matrix.DenseMatrix) *matrix.DenseMatrix {
	for i := 0; i < m1.Rows(); i++ {
		for j := 0; j < m1.Cols(); j++ {
			val := m1.Get(i, j)
			m1.Set(i, j, Sigmoid(val))
		}
	}

	return m1
}

// Sigmoid calculates the sigmoid of `z`
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// MatrixSigmoidPrime calculates the sigmoid prime of each element in `m1`
func MatrixSigmoidPrime(m1 *matrix.DenseMatrix) *matrix.DenseMatrix {
	for i := 0; i < m1.Rows(); i++ {
		for j := 0; j < m1.Cols(); j++ {
			val := m1.Get(i, j)
			m1.Set(i, j, SigmoidPrime(val))
		}
	}

	return m1
}

// SigmoidPrime calculates the sigmoid prime of `z`
func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}
