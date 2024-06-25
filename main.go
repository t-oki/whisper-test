package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials/stscreds"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type TranscriptionResult struct {
	Text string `json:"text"`
}

func main() {
	ctx := context.Background()

	audioFilePath := os.Args[1]
	awsProfile := os.Args[2]
	mfaToken := os.Args[3]

	awsConfig, err := config.LoadDefaultConfig(
		ctx,
		config.WithDefaultRegion("ap-northeast-1"),
		config.WithSharedConfigProfile(awsProfile),
		config.WithAssumeRoleCredentialOptions(func(o *stscreds.AssumeRoleOptions) {
			o.TokenProvider = func() (string, error) { return mfaToken, nil }
		}),
	)
	if err != nil {
		log.Fatal(err)
	}
	brc := bedrockruntime.NewFromConfig(awsConfig)

	transcription, err := transcribeAudio(audioFilePath)
	if err != nil {
		log.Fatal(err)
	}

	promptFile, err := os.Open("prompt.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer promptFile.Close()
	b, err := io.ReadAll(promptFile)
	if err != nil {
		log.Fatal(err)
	}

	prompt := fmt.Sprintf(`

Human: %s
%s

Assistant:`, b, transcription)

	payload := Request{Prompt: prompt, MaxTokensToSample: 4000}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}
	output, err := brc.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		Body:        []byte(payloadBytes),
		ModelId:     aws.String("anthropic.claude-instant-v1"),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		log.Fatal(err)
	}

	var resp Response

	err = json.Unmarshal(output.Body, &resp)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Response:", resp.Completion)
}

// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html
type Request struct {
	Prompt            string   `json:"prompt"`
	MaxTokensToSample int      `json:"max_tokens_to_sample"`
	Temperature       float64  `json:"temperature,omitempty"`
	TopP              float64  `json:"top_p,omitempty"`
	TopK              int      `json:"top_k,omitempty"`
	StopSequences     []string `json:"stop_sequences,omitempty"`
}

type Response struct {
	Completion string `json:"completion"`
}

func transcribeAudio(audioFilePath string) (string, error) {
	start := time.Now()

	cmd := exec.Command("python3", "whisper_script.py", audioFilePath)
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	fmt.Printf("Process time: %s\n", time.Since(start))

	var result TranscriptionResult
	if err := json.Unmarshal(output, &result); err != nil {
		return "", err
	}

	fmt.Println("Transcribed text:", result.Text)
	return result.Text, nil
}
