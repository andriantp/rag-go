package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
)

const (
	ollamaEmbedURL = "http://localhost:11434/api/embed"
	embeddingModel = "nomic-embed-text"
)

type embeddingRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type embeddingResponse struct {
	Embeddings [][]float64 `json:"embeddings"`
}

type semanticResult struct {
	text  string
	score float64
}

func embed(text string) ([]float64, error) {
	body, err := json.Marshal(embeddingRequest{
		Model: embeddingModel,
		Input: text,
	})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(
		http.MethodPost,
		ollamaEmbedURL,
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request embedding: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"Ollama embedding API returned status %s",
			resp.Status,
		)
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode embedding response: %w", err)
	}

	if len(result.Embeddings) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}

	return result.Embeddings[0], nil
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func semanticRetrieve(path, query string, topK int) (string, error) {

	return semanticRetrieveWithThreshold(
		path,
		query,
		topK,
		0.45,
	)
}

func semanticRetrieveWithThreshold(
	path, query string,
	topK int,
	minSimilarity float64,
) (string, error) {
	if strings.TrimSpace(query) == "" || topK <= 0 {
		return "", nil
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read knowledge file: %w", err)
	}

	paragraphs := splitParagraphs(string(data))
	if len(paragraphs) == 0 {
		return "", nil
	}

	queryEmbedding, err := embed(query)
	if err != nil {
		return "", fmt.Errorf("embed query: %w", err)
	}

	var results []semanticResult

	for _, paragraph := range paragraphs {
		paragraphEmbedding, err := embed(paragraph)
		if err != nil {
			return "", fmt.Errorf("embed paragraph: %w", err)
		}

		score := cosineSimilarity(queryEmbedding, paragraphEmbedding)

		if score < minSimilarity {
			continue
		}

		results = append(results, semanticResult{
			text:  paragraph,
			score: score,
		})
	}

	// Sort by similarity score, highest first.
	sort.SliceStable(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	// Log ranked results before applying topK.
	fmt.Printf("\nQuery: %s\n\n", query)
	fmt.Printf("%-6s %-10s %s\n", "Rank", "Score", "Chunk ID")

	for i, result := range results {
		fmt.Printf("%-6d %-10.4f %s\n",
			i+1,
			result.score,
			identifyChunk(result.text),
		)
	}

	if len(results) > topK {
		results = results[:topK]
	}

	var matches []string
	for _, result := range results {
		matches = append(matches, result.text)
	}

	return strings.Join(matches, "\n\n"), nil
}
