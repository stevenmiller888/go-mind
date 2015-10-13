package mind

import (
	"math"
	"math/rand"
	"time"

	"github.com/skelterjohn/go.matrix"
)

// Format the examples.
func Format(examples [][][]float64) (*matrix.DenseMatrix, *matrix.DenseMatrix) {
	var output, input [][]float64

	for _, example := range examples {
		output = append(output, example[1])
		input = append(input, example[0])
	}

	return matrix.MakeDenseMatrixStacked(input), matrix.MakeDenseMatrixStacked(output)
}

// Normals returns a DenseMatrix filled with random values.
func Normals(rows, cols int) *matrix.DenseMatrix {
	rand.Seed(time.Now().UTC().UnixNano())
	ret := matrix.Zeros(rows, cols)

	for i := 0; i < ret.Rows(); i++ {
		for j := 0; j < ret.Cols(); j++ {
			ret.Set(i, j, rand.NormFloat64())
		}
	}

	return ret
}

// Activator returns the activation function.
func Activator(f func(float64) float64) func(*matrix.DenseMatrix) *matrix.DenseMatrix {
	return func(m *matrix.DenseMatrix) *matrix.DenseMatrix {
		res := matrix.Zeros(m.Rows(), m.Cols())

		for i := 0; i < m.Rows(); i++ {
			for j := 0; j < m.Cols(); j++ {
				val := m.Get(i, j)
				res.Set(i, j, f(val))
			}
		}

		return res
	}
}

// Sigmoid calculates the sigmoid of `z`.
func Sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// SigmoidPrime calculates the sigmoid prime of `z`.
func SigmoidPrime(z float64) float64 {
	return Sigmoid(z) * (1 - Sigmoid(z))
}

// Htan calculates the hyperbolic tangent of `z`.
func Htan(z float64) float64 {
	return (math.Exp(2*z) - 1) / (math.Exp(2*z) + 1)
}

// Htanprime calculates the derivative of the hyperbolic tangent of `z`.
func Htanprime(z float64) float64 {
	return 1 - (math.Pow((math.Exp(2*z)-1)/(math.Exp(2*z)+1), 2))
}
