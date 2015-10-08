package mind

// Version
const Version = "0.0.1"

// Mind struct
type Mind struct {
	Activator    string  // activation function, `sigmoid` or `htan`
	LearningRate float64 // speed the network will learn at
	Iterations   int     // number of training iterations
	HiddenLayers int     // number of hidden layers
	HiddenUnits  int     // number of units in hidden layer/s
}

// New Mind
func New(activator string, rate float64, iterations int, layers int, units int) *Mind {
	m := &Mind{
		Activator:    activator,
		LearningRate: rate,
		Iterations:   iterations,
		HiddenLayers: layers,
		HiddenUnits:  units,
	}

	return m
}

// Learn from examples
func (m *Mind) Learn(examples [][]float64) {

}

// Predict from input
func (m *Mind) Predict() {

}
