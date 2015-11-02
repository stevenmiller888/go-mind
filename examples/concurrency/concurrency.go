package main

import (
	"fmt"

	"github.com/stevenmiller888/go-mind"
)

func main() {
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
