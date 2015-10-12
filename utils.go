package mind

import (
	"math"
	"math/rand"
	"time"

	"github.com/skelterjohn/go.matrix"
)

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
