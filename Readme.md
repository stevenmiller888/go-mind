# go-mind

A neural network library built in Go.

## Usage

```go
import "github.com/stevenmiller888/mind"

m := mind.New("sigmoid", 0.7, 10000, 2, 3)

m.Learn([][][]float64{
	{{0, 0}, {0}},
	{{0, 1}, {1}},
	{{1, 0}, {1}},
	{{1, 1}, {0}},
})

m.Predict([][]float64{
	{1, 0},
})
```

## Note

Just built this to learn a little Go :)