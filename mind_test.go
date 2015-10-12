package mind_test

import (
	"testing"

	"github.com/bmizerany/assert"
	"github.com/stevenmiller888/go-mind"
)

func TestMind(t *testing.T) {
	m := mind.New(0.7, 10000, 3)

	assert.Equal(t, m.LearningRate, 0.7)
	assert.Equal(t, m.Iterations, 10000)
	assert.Equal(t, m.HiddenUnits, 3)

	m.Learn([][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	})

	m.Predict([][]float64{
		{1, 0},
	})
}
