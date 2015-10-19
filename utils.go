package mind

import (
	"github.com/gonum/matrix/mat64"
	//"github.com/skelterjohn/go.matrix"
	//"fmt"
	"math"
	"math/rand"
	"time"
)

// Format the examples.
func Format(examples [][][]float64) (in *mat64.Dense, out *mat64.Dense) {
	/*fmt.Println("Length of examples: ", len(examples))
	fmt.Println("Length of items within examples: ", len(examples[0]))
	fmt.Println("Length of input of items within examples", len(examples[0][0]))
	fmt.Println("Length of output of items within examples", len(examples[0][1]))
	*/
	input := make([][]float64, len(examples[0])) //, len(examples[0][0]))
	for i := 0; i < len(input); i++ {
		input[i] = make([]float64, len(examples[0][0]))
	}

	output := make([][]float64, len(examples[0])) //, len(examples[0][1]))
	for i := 0; i < len(output); i++ {
		output[i] = make([]float64, len(examples[0][1]))
	}

	for _, example := range examples {
		//fmt.Println("length of example(in): ", len(example[0]))
		//fmt.Println("length of example(out): ", len(example[1]))
		input = append(input, example[0])
		output = append(output, example[1])
	}
	in = MakeDenseMatrixStacked(input)
	out = MakeDenseMatrixStacked(output)
	return in, out
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
	//for i := 0; i < len(input); i++ {
	//	fmt.Println("Row number: ", i, "; number of items in row: ", len(input[i]))
	//}
	//fmt.Println("# of rows:", len(input))
	//fmt.Println("# of columns: ", len(input[0]))
	r := len(input)
	c := len(input[0])
	mat := mat64.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		//for j := 0; j < c; j++ {
		//	//_, ok := input[r][c]
		//	//if ok == false {
		//	fmt.Printf("Out of bounds at row %d, column %d", i, j)
		//	mat.Set(i, j, input[i][j])
		//}
		//fmt.Println("row ", i, "; length: ", len(input[i]))
		mat.SetRow(i, input[i])
	}
	return mat
}
