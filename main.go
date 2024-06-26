package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials/stscreds"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	_ "github.com/joho/godotenv/autoload"
	"github.com/oklog/ulid/v2"
)

var m map[string]string = map[string]string{
	"Haiku":  "anthropic.claude-3-haiku-20240307-v1:0",
	"Sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
}

var audioPath string = "samples/"
var transcriptionPath string = "transcriptions/"

func main() {
	ctx := context.Background()

	mode := os.Args[1]  // download / transcribe / invoke
	model := os.Args[2] // Haiku / Sonnet
	mfaToken := os.Args[3]

	awsProfile := os.Getenv("AWS_PROFILE")
	s3BucketName := os.Getenv("S3_BUCKET_NAME")
	s3KeyPrefix := os.Getenv("S3_KEY_PREFIX")

	cfg, err := config.LoadDefaultConfig(
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
	s3Client := s3.NewFromConfig(cfg)
	cfg.Region = "us-west-2"
	brc := bedrockruntime.NewFromConfig(cfg)

	switch mode {
	case "download":
		fmt.Println("start download")
		if err := downloadObjects(ctx, s3Client, s3BucketName, s3KeyPrefix); err != nil {
			log.Fatal(err)
		}
		fmt.Println("finished download")
	case "transcribe":
		fmt.Println("start transcribe")
		if err := transcribe(); err != nil {
			log.Fatal(err)
		}
		fmt.Println("finished transcribe")
	case "invoke":
		fmt.Println("start invoke")
		if err := invoke(ctx, brc, model); err != nil {
			log.Fatal(err)
		}
		fmt.Println("finished invoke")
	default:
		fmt.Println("invalid mode")
		os.Exit(1)
	}

}

// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html
type Request struct {
	AnthropicVersion string           `json:"anthropic_version"`
	MaxTokens        int              `json:"max_tokens"`
	Messages         []RequestMessage `json:"messages"`
}

type RequestMessage struct {
	Role    string                  `json:"role"`
	Content []RequestMessageContent `json:"content"`
}

type RequestMessageContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type Response struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	StopReason   string `json:"stop_reason"`
	StopSequence string `json:"stop_sequence"`
	ToolUse      struct {
		Type  string `json:"type"`
		ID    string `json:"id"`
		Input struct {
		} `json:"input"`
	} `json:"tool_use"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func invoke(ctx context.Context, brc *bedrockruntime.Client, model string) error {
	fmt.Printf("=== Start %s ===", model)
	fmt.Println("")

	promptFile, err := os.Open("prompt.txt")
	if err != nil {
		return err
	}
	defer promptFile.Close()
	promptBytes, err := io.ReadAll(promptFile)
	if err != nil {
		return err
	}
	fmt.Println(string(promptBytes))

	transcriptionFiles, err := os.ReadDir(transcriptionPath)
	if err != nil {
		return err
	}

	for _, transcriptionFile := range transcriptionFiles {
		fmt.Println("=====================================")
		fmt.Println(transcriptionFile.Name())

		transcrptionFile, err := os.Open(transcriptionPath + transcriptionFile.Name())
		if err != nil {
			return err
		}
		defer transcrptionFile.Close()
		transcriptionBytes, err := io.ReadAll(transcrptionFile)
		if err != nil {
			return err
		}

		prompt := fmt.Sprintf(`%s
%s`, promptBytes, transcriptionBytes)
		fmt.Println(string(transcriptionBytes))
		payload := Request{
			AnthropicVersion: "bedrock-2023-05-31",
			MaxTokens:        1000,
			Messages: []RequestMessage{
				{
					Role: "user",
					Content: []RequestMessageContent{
						{
							Type: "text",
							Text: prompt,
						},
					},
				},
			},
		}
		payloadBytes, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		output, err := brc.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			Body:        []byte(payloadBytes),
			ModelId:     aws.String(m[model]),
			ContentType: aws.String("application/json"),
		})
		if err != nil {
			return err
		}

		var resp Response

		err = json.Unmarshal(output.Body, &resp)
		if err != nil {
			return err
		}

		fmt.Printf("=== %s Response ===", model)
		fmt.Println("")
		for _, v := range resp.Content {
			fmt.Println(v.Text)
		}
	}
	return nil
}

func transcribe() error {
	mp3s, err := os.ReadDir(audioPath)
	if err != nil {
		return err
	}

	for _, v := range mp3s {
		transcription, err := transcribeAudio(audioPath + v.Name())
		if err != nil {
			return err
		}

		file, err := os.Create(transcriptionPath + strings.Split(v.Name(), ".")[0] + ".txt")
		if err != nil {
			return err
		}
		defer file.Close()
		_, err = file.Write([]byte(transcription))
		if err != nil {
			return err
		}
	}
	return nil
}

type TranscriptionResult struct {
	Text string `json:"text"`
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

func downloadObjects(ctx context.Context, s3Client *s3.Client, s3BucketName, s3KeyPrefix string) error {
	objects, err := s3Client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket: aws.String(s3BucketName),
		Prefix: aws.String(s3KeyPrefix),
	})
	if err != nil {
		return err
	}

	for _, v := range objects.Contents {
		splittedFileName := strings.Split(*v.Key, ".")
		audioFileName := fmt.Sprintf("%s.%s", ulid.MustNew(ulid.Timestamp(time.Now()), nil), splittedFileName[len(splittedFileName)-1])
		if splittedFileName[len(splittedFileName)-1] != "mp3" {
			continue
		}

		object, err := s3Client.GetObject(ctx, &s3.GetObjectInput{
			Bucket: aws.String(s3BucketName),
			Key:    v.Key,
		})
		if err != nil {
			return err
		}
		defer object.Body.Close()

		file, err := os.Create(audioFileName)
		if err != nil {
			return err
		}
		defer file.Close()
		body, err := io.ReadAll(object.Body)
		if err != nil {
			return err
		}
		_, err = file.Write(body)
		if err != nil {
			return err
		}
	}
	return nil
}
