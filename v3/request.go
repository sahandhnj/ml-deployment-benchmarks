package main

import (
	"encoding/json"
	"fmt"
	"image"
	"image/draw"
	"image/jpeg"
	"io"
	"io/ioutil"
	"math/rand"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"golang.org/x/image/colornames"
)

type Request struct {
	ID        string
	ImageFile multipart.File
	ImageName string
	w         http.ResponseWriter
}

const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

func RandStringBytes(n int) string {
	b := make([]byte, n)
	for i := range b {
		b[i] = letterBytes[rand.Intn(len(letterBytes))]
	}
	return string(b)
}

func (p *Request) predict() error {
	fmt.Println("Handling request " + time.Now().String())
	modeldir := "model"
	labelfile := "labels.txt"

	defer p.ImageFile.Close()

	inputName := "input/" + p.ImageName + "-input-" + p.ID + ".jpg"
	outputName := "output/" + p.ImageName + "-output-" + p.ID + ".jpg"

	// Load the labels
	loadLabels(labelfile)

	// Load a frozen graph to use for queries
	modelpath := filepath.Join(modeldir, "frozen_inference_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		return err
	}

	inputFile, err := os.Create(inputName)

	if err != nil {
		return err
	}
	defer inputFile.Close()

	_, err = io.Copy(inputFile, p.ImageFile)
	if err != nil {
		return err
	}

	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, i, err := makeTensorFromImage(inputName)
	if err != nil {
		return err
	}
	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return err
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return err
	}
	defer session.Close()

	// Transform the decoded YCbCr JPG image into RGBA
	b := i.Bounds()
	img := image.NewRGBA(b)
	draw.Draw(img, b, i, b.Min, draw.Src)

	// Get all the input and output operations
	inputop := graph.Operation("image_tensor")
	// Output ops
	o1 := graph.Operation("detection_boxes")
	o2 := graph.Operation("detection_scores")
	o3 := graph.Operation("detection_classes")
	o4 := graph.Operation("num_detections")

	// Execute COCO Graph
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputop.Output(0): tensor,
		},
		[]tf.Output{
			o1.Output(0),
			o2.Output(0),
			o3.Output(0),
			o4.Output(0),
		},
		nil)
	if err != nil {
		return err
	}

	// Outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Draw a box around the objects
	curObj := 0

	// 0.4 is an arbitrary threshold, below this the results get a bit random
	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		Rect(img, int(x1), int(y1), int(x2), int(y2), 4, colornames.Map[colornames.Names[int(classes[curObj])]])
		addLabel(img, int(x1), int(y1), int(classes[curObj]), getLabel(curObj, probabilities, classes))

		curObj++
	}

	// Output JPG file
	outfile, err := os.Create(outputName)
	if err != nil {
		return err
	}

	var opt jpeg.Options

	opt.Quality = 80

	err = jpeg.Encode(outfile, img, &opt)
	if err != nil {
		return err
	}

	done = done + 1
	fmt.Printf("Finished predicting #%d with Id:%s\n", done, p.ID)

	if !QueuedResult {
		js, err := json.Marshal(&presponse{ID: p.ID})
		if err != nil {
			return err
		}
		p.w.Header().Set("Content-Type", "application/json")
		p.w.Write(js)
	}

	return nil
}
