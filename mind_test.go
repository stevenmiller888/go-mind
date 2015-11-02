package mind_test

import (
	"fmt"
	"testing"

	"github.com/bmizerany/assert"
	"github.com/stevenmiller888/go-mind"
)

func TestMind(t *testing.T) {
	m := mind.New(0.3, 10000, 3, "sigmoid")

	assert.Equal(t, m.LearningRate, 0.3)
	assert.Equal(t, m.Iterations, 10000)
	assert.Equal(t, m.HiddenUnits, 3)

	m.Learn([][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	})

	prediction := m.Predict([][]float64{
		{0, 1},
	})

	fmt.Println(prediction.At(0, 0))
}

func TestMindConcurrent(t *testing.T) {
	c := make(chan float64)

	for i := 0; i < 10; i++ {
		go func() {
			m := mind.New(0.3, 10000, 3, "htan")

			m.Learn([][][]float64{
				{{0, 0}, {0}},
				{{0, 1}, {1}},
				{{1, 0}, {1}},
				{{1, 1}, {0}},
			})

			prediction := m.Predict([][]float64{
				{0, 1},
			})

			c <- prediction.At(0, 0)
		}()
	}

	for i := 0; i < 10; i++ {
		fmt.Println(<-c)
	}
}
