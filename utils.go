package mind

import (
	"github.com/gonum/matrix/mat64"
	//"github.com/skelterjohn/go.matrix"
	"math"
	"math/rand"
	"time"
)

// Format the examples.
func Format(examples [][][]float64) (*mat64.Dense, *mat64.Dense) {
	var output, input [][]float64
	for _, example := range examples {
		output = append(output, example[1])
		input = append(input, example[0])
	}

	return MakeDenseMatrixStacked(input), MakeDenseMatrixStacked(output)
}

// Normals returns a mat64.DenseMatrix filled with random values.
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
		r, c := m.Dims()
		res := mat64.NewDense(r, c, nil)
		for i := 0; i < r; i++ {
			for j := 0; j < c; j++ {
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

func MakeDenseMatrixStacked(input [][]float64) *mat64.Dense {
	r := len(input)
	c := len(input[0])
	mat := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			mat.Set(i, j, input[r][c])
		}
	}
	return mat
}
