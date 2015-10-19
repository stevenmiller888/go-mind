package main

import (
	"github.com/vibbix/go-mind"
	"image"
	_ "image/png"
	"log"
	"os"
)

var (
	A = ReadImage("A")
	B = ReadImage("B")
	C = ReadImage("C")
	D = ReadImage("D")
)

func main() {
	m := mind.New(0.3, 100, 2, "sigmoid")
	m.Learn([][][]float64{
		{A, mapletter('A')},
		{B, mapletter('B')},
		{C, mapletter('C')},
		{D, mapletter('D')},
	})
	result := m.Predict([][]float64{C})
	log.Println(result) //should be around .50 or so
}

func mapletter(letter byte) []float64 {
	if letter == 'A' {
		return []float64{0.1}
	}
	if letter == 'B' {
		return []float64{0.3}
	}
	if letter == 'C' {
		return []float64{0.5}
	}
	if letter == 'D' {
		return []float64{0.7}
	}
	return []float64{0.0}

}

func ReadImage(letter string) []float64 {
	file, err := os.Open(letter + ".png") //could be done better with path.join
	if err != nil {
		log.Fatal("Couldn't open file of letter " + letter)
	}
	defer file.Close()
	//imgread := base64.NewDecoder(base64.StdEncoding, file)
	img, _, imgerr := image.Decode(file)
	if imgerr != nil {
		log.Fatal("Couldn't parse image letter " + letter + "; " + imgerr.Error())
	}
	bounds := img.Bounds()
	flt := make([]float64, ((bounds.Max.Y - bounds.Min.Y) * (bounds.Max.X - bounds.Min.X) * 4))
	//log.Println("letter ", letter, " info: max-", bounds.Max, " ; min-", bounds.Min)
	i := -1 //starts at negative, to iterate
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			i++
			flt[i] = (float64(r) / 255.0)
			i++
			flt[i] = (float64(g) / 255.0)
			i++
			flt[i] = (float64(b) / 255.0)
			i++
			flt[i] = (float64(a) / 255.0)
		}
	}
	//log.Println("size of letter ", letter, ":", len(flt))
	return flt
}
