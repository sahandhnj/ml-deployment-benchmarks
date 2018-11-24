package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

var works int = 0

type presponse struct {
	ID string `json:"id"`
}

const (
	defaultMaxMemory2 = 32 << 20 // 32 MB
)

func requestHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println(formatRequest(r))
	if r.Method != "POST" {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	r.ParseMultipartForm(defaultMaxMemory2)
	fmt.Println(r.MultipartForm)

	imageFile, header, err := r.FormFile("file")

	if err != nil {
		fmt.Println("Could not read image", err)
		return
	}
	imageName := strings.Split(header.Filename, ".")

	id := RandStringBytes(10)

	work := Job{Request: Request{
		ID:        id,
		ImageFile: imageFile,
		ImageName: imageName[0],
	}}
	works = works + 1

	JobQueue <- work
	fmt.Printf("<- work #%d, queue size: %d\n", works, len(JobQueue))

	js, err := json.Marshal(&presponse{ID: id})
	if err != nil {
		fmt.Println(err)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Write(js)
}

func formatRequest(r *http.Request) string {
	// Create return string
	var request []string
	// Add the request string
	url := fmt.Sprintf("%v %v %v", r.Method, r.URL, r.Proto)
	request = append(request, url)
	// Add the host
	request = append(request, fmt.Sprintf("Host: %v", r.Host))
	// Loop through headers
	for name, headers := range r.Header {
		name = strings.ToLower(name)
		for _, h := range headers {
			request = append(request, fmt.Sprintf("%v: %v", name, h))
		}
	}

	// If this is a POST, add post data
	if r.Method == "POST" {
		r.ParseForm()
		request = append(request, "\nBody:\n")
		request = append(request, r.Form.Encode())
	}
	// Return the request as a string
	return strings.Join(request, "\n")
}
