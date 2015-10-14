package main

import (
	"github.com/stevenmiller888/go-mind"
	"log"
)

func main() {
	m := mind.New(0.3, 10000, 3, "sigmoid")

	m.Learn([][][]float64{
		{{0, 0}, {0}},
		{{0, 1}, {1}},
		{{1, 0}, {1}},
		{{1, 1}, {0}},
	})

	result := m.Predict([][]float64{
		{0, 1},
	})
	log.Println(result) //should be close to 1.0
}
