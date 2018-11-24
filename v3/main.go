package main

import (
	"fmt"
	"log"
	"net/http"
)

const (
	MaxWorker = 2
	MaxQueue  = 50
	Address   = ":3002"
)

func main() {
	http.HandleFunc("/predict", requestHandler)

	fmt.Printf("size of queue %d\n", MaxQueue)
	JobQueue = make(chan Job, MaxQueue)

	dispatcher := NewDispatcher(MaxWorker)
	dispatcher.Run()

	fmt.Println("Listening on " + Address)
	err := http.ListenAndServe(Address, nil)
	if err != nil {
		log.Fatal(err)
	}
}
