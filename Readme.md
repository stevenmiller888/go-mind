[![](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner2-direct.svg)](https://github.com/vshymanskyy/StandWithUkraine/blob/main/docs/README.md)

# go-mind

A neural network library built in Go.

## Usage

```go
import "github.com/stevenmiller888/go-mind"

m := mind.New(0.7, 10000, 3, "sigmoid")

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

Just built this to learn a little Go. Feedback welcome :)

## License

MIT
