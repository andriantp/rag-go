package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/prompts"
)

// var model = "deepseek-r1:1.5b"
// var model = "gemma3:4b"
var model = "qwen2.5:3b"

func main() {
	if len(os.Args) < 2 {
		log.Fatal("usage: go run . <BASIC|INTERACTION|TEMPLATE|CHAIN|JSON|RETRIEVAL|SEMANTIC|RAG|EVALUATE>")
	}

	ctx := context.Background()
	mode := strings.ToUpper(os.Args[1])

	options := []ollama.Option{
		ollama.WithModel(model),
	}

	if mode == "JSON" {
		options = append(options, ollama.WithFormat("json"))
	}

	fmt.Println("Using model:", model)
	llm, err := ollama.New(options...)
	if err != nil {
		log.Fatal(err)
	}

	var prompt string

	switch mode {
	case "BASIC":
		prompt = "Explain what RAG is in simple terms."

	case "INTERACTION":
		fmt.Print("You: ")

		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				log.Fatal(err)
			}
			log.Fatal("no input provided")
		}

		prompt = scanner.Text()

	case "TEMPLATE":
		template := prompts.NewPromptTemplate(
			"Answer the following question clearly and concisely: {{.question}}",
			[]string{"question"},
		)

		prompt, err = template.Format(map[string]any{
			"question": "What is RAG?",
		})
		if err != nil {
			log.Fatal(err)
		}

	case "CHAIN":
		template := prompts.NewPromptTemplate(
			"Answer the following question clearly and concisely: {{.question}}",
			[]string{"question"},
		)

		chain := chains.NewLLMChain(llm, template)

		response, err := chains.Run(
			ctx,
			chain,
			"What is RAG?",
		)
		if err != nil {
			log.Fatal(err)
		}

		fmt.Println(response)
		return

	case "JSON":
		template := prompts.NewPromptTemplate(
			`Answer the question using the following facts:
			- RAG stands for Retrieval-Augmented Generation.
			- It retrieves relevant external information and provides it as context to an LLM.
			- Its purpose is to ground generated answers in external information.

			Return exactly one JSON object with these fields:
			- topic: string
			- definition: string
			- purpose: string

			Question: {{.question}}`,
			[]string{"question"},
		)

		prompt, err = template.Format(map[string]any{
			"question": "What is RAG?",
		})
		if err != nil {
			log.Fatal(err)
		}

	case "RETRIEVAL", "SEMANTIC":
		fmt.Print("You: ")

		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				log.Fatal(err)
			}
			log.Fatal("failed to read input")
		}

		question := scanner.Text()
		var retrievedContext string
		var err error

		switch mode {
		case "RETRIEVAL":
			retrievedContext, err = retrieve(
				"data/knowledge.txt",
				question,
				3,
			)
		case "SEMANTIC":
			retrievedContext, err = semanticRetrieve(
				"data/knowledge.txt",
				question,
				3,
			)
		}

		if err != nil {
			log.Fatal(err)
		}

		if retrievedContext == "" {
			fmt.Println("No relevant information found.")
			return
		}

		fmt.Println("\nRetrieved Context:")
		fmt.Println(retrievedContext)
		fmt.Println("--------------------")

		template := prompts.NewPromptTemplate(
			`Answer the question using only the information explicitly stated in the context.

			Rules:
			- Do not add assumptions, examples, or external knowledge.
			- If the context does not contain enough information, say: "The provided context does not contain enough information to answer this question."
			- Keep the answer concise.

			Context:
			{{.context}}

			Question: {{.question}}

			Answer:`,
			[]string{"context", "question"},
		)

		prompt, err = template.Format(map[string]any{
			"context":  retrievedContext,
			"question": question,
		})
		if err != nil {
			log.Fatal(err)
		}

	case "EVALUATE":
		evalFlags := flag.NewFlagSet("EVALUATE", flag.ExitOnError)
		
		topK := evalFlags.Int(
			"topk",
			2,
			"maximum number of retrieved paragraphs",
		)

		minSimilarity := evalFlags.Float64(
			"min-similarity",
			0.45,
			"minimum cosine similarity for semantic retrieval",
		)

		if err := evalFlags.Parse(os.Args[2:]); err != nil {
			log.Fatal(err)
		}

		if *topK < 1 {
			log.Fatal("topk must be greater than 0")
		}

		if *minSimilarity < -1 || *minSimilarity > 1 {
			log.Fatal("min-similarity must be between -1 and 1")
		}

		if err := runEvaluation(*topK, *minSimilarity); err != nil {
			log.Fatal(err)
		}

		return

	case "RAG":
		fmt.Print("You: ")

		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				log.Fatal(err)
			}
			log.Fatal("failed to read input")
		}

		question := scanner.Text()

		retrievedContext, err := semanticRetrieveWithThreshold(
			"data/knowledge.txt",
			question,
			2,
			0.45,
		)
		if err != nil {
			log.Fatal(err)
		}

		if retrievedContext == "" {
			fmt.Println("No relevant information found.")
			return
		}

		fmt.Println("\nRetrieved Context:")
		fmt.Println(retrievedContext)
		fmt.Println("--------------------")

		template := prompts.NewPromptTemplate(
			`Answer the question using only the information explicitly stated in the context.

			Rules:
			- Every factual claim must be explicitly supported by the context.
			- Do not infer mechanisms or add details that are not explicitly stated in the context.
			- If a detail is not stated in the context, do not include it in the answer.
			- If the context does not contain enough information, say: "The provided context does not contain enough information to answer this question."
			- Keep the answer concise.
			- Answer the specific question directly using the most relevant facts from the context.
			- Preserve important qualifications and limitations stated in the context.

			Context:
			{{.context}}

			Question: {{.question}}

			Answer:`,
			[]string{"context", "question"},
		)

		prompt, err = template.Format(map[string]any{
			"context":  retrievedContext,
			"question": question,
		})
		if err != nil {
			log.Fatal(err)
		}

	default:
		log.Fatalf("UNKNOWN ARGUMENT: %s", os.Args[1])
	}

	response, err := llms.GenerateFromSinglePrompt(ctx, llm, prompt)
	if err != nil {
		log.Fatal(err)
	}

	if mode != "JSON" {
		fmt.Println(response)
		return
	}

	var result struct {
		Topic      string `json:"topic"`
		Definition string `json:"definition"`
		Purpose    string `json:"purpose"`
	}

	if err := json.Unmarshal([]byte(response), &result); err != nil {
		log.Fatalf("invalid JSON response: %v\nResponse: %s", err, response)
	}

	if result.Topic == "" || result.Definition == "" || result.Purpose == "" {
		log.Fatalf("missing required fields: %+v\nResponse: %s", result, response)
	}

	fmt.Printf("Topic: %s\n", result.Topic)
	fmt.Printf("Definition: %s\n", result.Definition)
	fmt.Printf("Purpose: %s\n", result.Purpose)
}
