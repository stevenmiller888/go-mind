package mind

import (
	"math"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
)

// Format the examples.
func Format(examples [][][]float64) (*mat64.Dense, *mat64.Dense) {
	var input, output []float64
	rows := len(examples)
	inCols := len(examples[0][0])
	outCols := len(examples[0][1])

	for _, example := range examples {
		output = append(output, example[1]...)
		input = append(input, example[0]...)
	}

	return mat64.NewDense(rows, inCols, input), mat64.NewDense(rows, outCols, output)
}

// Normals returns a DenseMatrix filled with random values.
func Normals(rows, cols int) *mat64.Dense {
	rand.Seed(time.Now().UTC().UnixNano())
	ret := mat64.NewDense(rows, cols, nil)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			ret.Set(i, j, rand.NormFloat64())
		}
	}

	return ret
}

// Activator returns the activation function.
func Activator(f func(float64) float64) func(*mat64.Dense) *mat64.Dense {
	return func(m *mat64.Dense) *mat64.Dense {
		rows, cols := m.Dims()
		res := mat64.NewDense(rows, cols, nil)

		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := m.At(i, j)
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
