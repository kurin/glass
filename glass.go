package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/ioutil"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"time"

	kdtree "github.com/hongshibao/go-kdtree"
)

type line struct {
	a, b, c float64
}

type point struct {
	vec   []float64
	color color.Color
}

type img interface {
	Set(int, int, color.Color)
}

func (p *point) draw(img img, c color.Color) {
	img.Set(int(p.vec[0]), int(p.vec[1]), c)
}

func (p *point) isIn(ps []kdtree.Point) bool {
	for _, pp := range ps {
		if reflect.DeepEqual(p, pp) {
			return true
		}
	}
	return false
}

func (p *point) line(o *point) line {
	return line{
		a: p.vec[1] - o.vec[1],                   // y0 - y1
		b: o.vec[0] - p.vec[0],                   // x1 - x0
		c: p.vec[1]*o.vec[0] - o.vec[1]*p.vec[0], // y0x1 - y1x0
	}
}

func (l line) perp(p *point) line {
	return line{
		a: l.b,
		b: -l.a,
		c: l.b*p.vec[0] - l.a*p.vec[1],
	}
}

func (p *point) biscector(o *point) line {
	var mps []float64
	for i := 0; i < p.Dim(); i++ {
		mps = append(mps, (p.vec[i]+o.vec[i])/2)
	}
	mp := &point{vec: mps}
	return p.line(o).perp(mp)
}

func (p *point) xy() xy {
	return xy{x: p.vec[0], y: p.vec[1]}
}

func (l line) draw(img img, c color.Color) {
	for x := 0; x < maxX; x++ {
		y := -(l.a/l.b)*float64(x) + l.c/l.b
		img.Set(int(x), int(y), c)
	}
	for y := 0; y < maxY; y++ {
		x := -(l.b/l.a)*float64(y) + l.c/l.a
		img.Set(int(x), int(y), c)
	}
}

type xy struct{ x, y float64 }

func (p xy) point() *point { return &point{vec: []float64{p.x, p.y}} }

type adjList struct {
	l map[xy]map[xy]bool
}

func (l *adjList) link(a, b kdtree.Point) {
	aa, bb := a.(*point), b.(*point)
	l.add(aa, bb)
	l.add(bb, aa)
}

func (l *adjList) add(a, b *point) {
	if l.l[a.xy()] == nil {
		l.l[a.xy()] = make(map[xy]bool)
	}
	l.l[a.xy()][b.xy()] = true
}

func (l line) drawNear(img img, tree *kdtree.KDTree, c color.Color, al *adjList) {
	for x := 0; x < maxX; x++ {
		y := -(l.a/l.b)*float64(x) + l.c/l.b
		n := &point{vec: []float64{float64(x), y}}
		nns := tree.KNN(n, 2)
		n0, n1 := nns[0], nns[1]
		if math.Abs(n.Distance(n0)-n.Distance(n1)) < 1 {
			al.link(n0, n1)
			img.Set(int(x), int(y), c)
		}
	}
	for y := 0; y < maxY; y++ {
		x := -(l.b/l.a)*float64(y) + l.c/l.a
		n := &point{vec: []float64{x, float64(y)}}
		nns := tree.KNN(n, 2)
		n0, n1 := nns[0], nns[1]
		if math.Abs(n.Distance(n0)-n.Distance(n1)) < 1 {
			al.link(n0, n1)
			img.Set(int(x), int(y), c)
		}
	}
}

func (p *point) Dim() int               { return len(p.vec) }
func (p *point) GetValue(d int) float64 { return p.vec[d] }

func (p *point) Distance(o kdtree.Point) float64 {
	var ret float64
	for i := 0; i < p.Dim(); i++ {
		tmp := p.GetValue(i) - o.GetValue(i)
		ret += tmp * tmp
	}
	return ret
}

func (p *point) PlaneDistance(val float64, dim int) float64 {
	tmp := p.GetValue(dim) - val
	return tmp * tmp
}

var (
	maxX      = 58 * 40
	maxY      = 20 * 40
	numPoints = flag.Int("num_points", 20, "number of points")
	seed      = flag.Int64("seed", time.Now().UnixNano(), "rng seed")
)

func main() {
	flag.Parse()
	rand.Seed(*seed)
	td, err := ioutil.TempDir("", "")
	if err != nil {
		fmt.Println(err)
		return
	}
	defer os.RemoveAll(td)

	var points []kdtree.Point
	for len(points) < *numPoints {
		x, y := rand.Float64(), rand.Float64()
		//if x > math.Pow(rand.Float64(), 2) {
		//	continue
		//}
		points = append(points, &point{
			vec:   []float64{x * float64(maxX), y * float64(maxY)},
			color: color.RGBA{0, 0, 0, 255},
		})
	}

	tree := kdtree.NewKDTree(points)
	img := image.NewRGBA(image.Rect(0, 0, maxX, maxY))

	al := &adjList{
		l: make(map[xy]map[xy]bool),
	}

	for i, p := range points {
		fmt.Println(i)
		for _, nn := range points[i+1:] {
			p.(*point).biscector(nn.(*point)).drawNear(img, tree, color.RGBA{0, 0, 0, 255}, al)
		}
	}

	colorPoints(tree, points, al)

	for x := 0; x < maxX; x++ {
		fmt.Printf("%d/%d\n", x, maxX)
		for y := 0; y < maxY; y++ {
			p := &point{vec: []float64{float64(x), float64(y)}}
			nn := tree.KNN(p, 1)[0].(*point)
			if _, _, _, a := img.At(x, y).RGBA(); a == 0 {
				img.Set(x, y, nn.color)
			}
		}
	}

	for x := 0; x < maxX; x += (maxX / 58) {
		for y := 0; y < maxY; y++ {
			img.Set(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	for y := 0; y < maxY; y += (maxY / 20) {
		for x := 0; x < maxX; x++ {
			img.Set(x, y, color.RGBA{128, 128, 128, 255})
		}
	}

	f, err := os.Create(filepath.Join(td, "image.png"))
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := png.Encode(f, img); err != nil {
		fmt.Println(err)
		return
	}
	if err := f.Close(); err != nil {
		fmt.Println(err)
		return
	}

	http.Handle("/", http.FileServer(http.Dir(td)))
	fmt.Println("ok; seed is", *seed, "count is", *numPoints)
	http.ListenAndServe(":8822", nil)
}

func (l *adjList) enStack(seen map[xy]bool) []xy {
	var out []xy
	for k, v := range l.l {
		var deg int
		for n := range v {
			if !seen[n] {
				deg++
			}
		}
		if deg < 6 && !seen[k] {
			seen[k] = true
			out = append(out, k)
			if len(seen) == len(l.l) {
				return out
			}
			return append(out, l.enStack(seen)...)
		}
	}
	for k := range l.l {
		seen[k] = true
		out = append(out, k)
		if len(seen) == len(l.l) {
			return out
		}
		return append(out, l.enStack(seen)...)
	}
	return nil
}

func colorPoints(tree *kdtree.KDTree, points []kdtree.Point, al *adjList) {
	stack := al.enStack(map[xy]bool{})
	for i := len(stack) - 1; i >= 0; i-- {
		p := tree.KNN(stack[i].point(), 1)[0].(*point)
		seen := map[color.Color]bool{}
		for n := range al.l[p.xy()] {
			seen[tree.KNN(n.point(), 1)[0].(*point).color] = true
		}
		rand.Shuffle(len(colors), func(i, j int) {
			colors[i], colors[j] = colors[j], colors[i]
		})
		for _, c := range colors {
			if !seen[c] {
				p.color = c
			}
		}
	}
}

var colors = []color.Color{
	color.RGBA{155, 17, 30, 255},
	color.RGBA{190, 83, 28, 255},
	color.RGBA{241, 196, 0, 255},
	color.RGBA{19, 104, 67, 255},
	color.RGBA{135, 206, 235, 255},
	color.RGBA{89, 49, 95, 255},
}
