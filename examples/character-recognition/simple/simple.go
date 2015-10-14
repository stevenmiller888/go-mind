package main

import (
	"github.com/stevenmiller888/go-mind"
	"log"
)

var (
	a = character(
		".#####." +
			"#.....#" +
			"#.....#" +
			"#######" +
			"#.....#" +
			"#.....#" +
			"#.....#")
	b = character(
		"######." +
			"#.....#" +
			"#.....#" +
			"######." +
			"#.....#" +
			"#.....#" +
			"######.")
	c = character(
		"#######" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#......" +
			"#######")
)

func main() {
	m := mind.New(0.3, 10000, 3, "sigmoid")
	m.Learn([][][]float64{
		{c, mapletter('c')},
		{b, mapletter('b')},
		{a, mapletter('a')},
	})

	result := m.Predict([][]float64{
		character(
			"#######" +
				"#......" +
				"#......" +
				"#......" +
				"#......" +
				"##....." +
				"#######")})
	log.Println(result)
	//Result should be somewhere around .5
}
func character(chars string) []float64 {
	flt := make([]float64, len(chars))
	for i := 0; i < len(chars); i++ {
		if chars[i] == '#' {
			flt[i] = 1.0
		} else { // if '.'
			flt[i] = 0.0
		}
	}
	return flt
}

func mapletter(letter byte) []float64 {
	if letter == 'a' {
		return []float64{0.1}
	}
	if letter == 'b' {
		return []float64{0.3}
	}
	if letter == 'c' {
		return []float64{0.5}
	}
	return []float64{0.0}
}
