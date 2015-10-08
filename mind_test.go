package mind_test

import (
	"testing"

	"github.com/bmizerany/assert"
	"github.com/stevenmiller888/go-mind"
)

func TestMind(t *testing.T) {
	m := mind.New("sigmoid", 0.7, 10000, 2, 3)

	assert.Equal(t, m.Activator, "sigmoid")
	assert.Equal(t, m.LearningRate, 0.7)
	assert.Equal(t, m.Iterations, 10000)
	assert.Equal(t, m.HiddenLayers, 2)
	assert.Equal(t, m.HiddenUnits, 3)
}
