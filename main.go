package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os/exec"
	"time"
)

type TranscriptionResult struct {
	Text string `json:"text"`
}

func main() {
	audioFilePath := "./sample.mp3"

	start := time.Now()

	cmd := exec.Command("python3", "whisper_script.py", audioFilePath)
	output, err := cmd.Output()
	if err != nil {
		log.Fatalf("Error running Python script: %v", err)
	}

	fmt.Printf("Process time: %s\n", time.Since(start))

	var result TranscriptionResult
	if err := json.Unmarshal(output, &result); err != nil {
		log.Fatalf("Error parsing JSON: %v", err)
	}

	fmt.Println("Transcribed text:", result.Text)
}
