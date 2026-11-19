package main

import (
	"fmt"
	"os"
	"strings"
)

type evalCase struct {
	name        string
	query       string
	expectedIDs []string
	outOfScope  bool
}

type evalMetrics struct {
	precisionSum float64
	recallSum    float64
	evaluated    int
	rejected     int
	outOfScope   int
}

var evalCases = []evalCase{
	// RAG
	{
		name:        "Direct RAG question",
		query:       "What does RAG stand for?",
		expectedIDs: []string{"rag"},
	},
	{
		name:        "RAG paraphrase",
		query:       "How can an LLM access external knowledge?",
		expectedIDs: []string{"rag"},
	},
	{
		name:        "RAG different wording",
		query:       "What technique helps language models answer using information from documents?",
		expectedIDs: []string{"rag"},
	},

	// Embedding
	{
		name:        "Embedding definition",
		query:       "What is an embedding?",
		expectedIDs: []string{"embedding"},
	},
	{
		name:        "Embedding paraphrase",
		query:       "How can text be represented numerically?",
		expectedIDs: []string{"embedding"},
	},

	// Retrieval
	{
		name:        "Retrieval definition",
		query:       "What is retrieval in a document search system?",
		expectedIDs: []string{"retrieval"},
	},
	{
		name:        "Retrieval paraphrase",
		query:       "How does a system find relevant documents for a query?",
		expectedIDs: []string{"retrieval"},
	},

	// Generation
	{
		name:        "Generation definition",
		query:       "What does generation mean in a language model?",
		expectedIDs: []string{"generation"},
	},
	{
		name:        "Generation in RAG",
		query:       "How does an LLM produce an answer using retrieved context?",
		expectedIDs: []string{"generation"},
	},

	// Vector Search
	{
		name:        "Vector search definition",
		query:       "What is vector search?",
		expectedIDs: []string{"vector_search"},
	},
	{
		name:        "Vector search paraphrase",
		query:       "How can a system find items with similar vector representations?",
		expectedIDs: []string{"vector_search"},
	},

	// Cosine Similarity
	{
		name:        "Cosine similarity definition",
		query:       "What does cosine similarity measure?",
		expectedIDs: []string{"cosine_similarity"},
	},
	{
		name:        "Cosine similarity range",
		query:       "What is the possible range of cosine similarity scores?",
		expectedIDs: []string{"cosine_similarity"},
	},

	// Chunking
	{
		name:        "Chunking definition",
		query:       "What is chunking in RAG?",
		expectedIDs: []string{"chunking"},
	},
	{
		name:        "Chunking purpose",
		query:       "Why divide a long document into smaller sections?",
		expectedIDs: []string{"chunking"},
	},

	// Top-K Retrieval
	{
		name:        "Top-K definition",
		query:       "What does topK control in retrieval?",
		expectedIDs: []string{"top_k"},
	},
	{
		name:        "Top-K paraphrase",
		query:       "How can I limit the number of retrieved chunks?",
		expectedIDs: []string{"top_k"},
	},

	// Similarity Threshold
	{
		name:        "Threshold definition",
		query:       "What is a similarity threshold?",
		expectedIDs: []string{"threshold"},
	},
	{
		name:        "Threshold behavior",
		query:       "How can low-scoring retrieval results be excluded?",
		expectedIDs: []string{"threshold"},
	},

	// Hallucination
	{
		name:        "Hallucination definition",
		query:       "What is an LLM hallucination?",
		expectedIDs: []string{"hallucination"},
	},
	{
		name:        "Hallucination prevention",
		query:       "Why can a model generate unsupported information even when using RAG?",
		expectedIDs: []string{"hallucination"},
	},

	// Keyword Search
	{
		name:        "Keyword search definition",
		query:       "How does keyword search retrieve documents?",
		expectedIDs: []string{"keyword_search"},
	},
	{
		name:        "Keyword search limitation",
		query:       "Why can exact term matching miss relevant information?",
		expectedIDs: []string{"keyword_search"},
	},

	// pgvector
	{
		name:        "pgvector definition",
		query:       "What is pgvector?",
		expectedIDs: []string{"pgvector"},
	},
	{
		name:        "pgvector use case",
		query:       "How can PostgreSQL store and search embeddings?",
		expectedIDs: []string{"pgvector"},
	},

	// Similar concepts
	{
		name:        "Embedding versus vector search",
		query:       "What represents the meaning of text, rather than searching for similar items?",
		expectedIDs: []string{"embedding"},
	},
	{
		name:        "Vector search versus keyword search",
		query:       "How can documents be found by semantic similarity instead of matching exact words?",
		expectedIDs: []string{"vector_search"},
	},

	// Out of scope
	{
		name:       "Out of scope Kubernetes",
		query:      "How do I configure a Kubernetes cluster?",
		outOfScope: true,
	},
	{
		name:       "Out of scope cooking",
		query:      "How do I bake sourdough bread?",
		outOfScope: true,
	},
}

// Identify a retrieved paragraph by its stable document heading.
// This function identifies chunks, not their relevance to a query.
func identifyChunk(paragraph string) string {
	text := strings.ToLower(strings.TrimSpace(paragraph))

	switch {
	case strings.HasPrefix(text, "rag (retrieval-augmented generation)"):
		return "rag"
	case strings.HasPrefix(text, "embedding"):
		return "embedding"
	case strings.HasPrefix(text, "retrieval"):
		return "retrieval"
	case strings.HasPrefix(text, "generation"):
		return "generation"
	case strings.HasPrefix(text, "vector search"):
		return "vector_search"
	case strings.HasPrefix(text, "cosine similarity"):
		return "cosine_similarity"
	case strings.HasPrefix(text, "chunking"):
		return "chunking"
	case strings.HasPrefix(text, "top-k retrieval"):
		return "top_k"
	case strings.HasPrefix(text, "similarity threshold"):
		return "threshold"
	case strings.HasPrefix(text, "hallucination"):
		return "hallucination"
	case strings.HasPrefix(text, "keyword search"):
		return "keyword_search"
	case strings.HasPrefix(text, "pgvector"):
		return "pgvector"
	default:
		return "unknown"
	}
}

func containsID(ids []string, target string) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}
	return false
}

func evaluateMethod(
	name string,
	retrieveFn func(string, string, int) (string, error),
	topK int,
) error {
	const dataPath = "data/knowledge.txt"

	var metrics evalMetrics

	fmt.Printf("\n========== %s ==========\n", name)

	for _, test := range evalCases {
		context, err := retrieveFn(dataPath, test.query, topK)
		if err != nil {
			return fmt.Errorf("%s: %w", test.name, err)
		}

		paragraphs := splitParagraphs(context)

		retrievedIDs := make(map[string]bool)
		relevantIDs := make(map[string]bool)
		irrelevantIDs := make(map[string]bool)

		for _, id := range test.expectedIDs {
			relevantIDs[id] = false
		}

		for _, paragraph := range paragraphs {
			id := identifyChunk(paragraph)

			if id == "unknown" {
				irrelevantIDs[id] = true
				continue
			}

			retrievedIDs[id] = true

			if _, expected := relevantIDs[id]; expected {
				relevantIDs[id] = true
			} else {
				irrelevantIDs[id] = true
			}
		}

		truePositives := 0
		missedIDs := make([]string, 0)

		for id, found := range relevantIDs {
			if found {
				truePositives++
			} else {
				missedIDs = append(missedIDs, id)
			}
		}

		precision := 0.0
		if len(paragraphs) > 0 {
			precision = float64(truePositives) /
				float64(len(paragraphs))
		}

		recall := 0.0
		if len(test.expectedIDs) > 0 {
			recall = float64(truePositives) /
				float64(len(test.expectedIDs))
		}

		fmt.Printf("\nTest: %s\n", test.name)
		fmt.Printf("Query: %s\n", test.query)
		fmt.Printf("Expected IDs: %v\n", test.expectedIDs)
		fmt.Printf("Retrieved IDs: %v\n", retrievedIDs)
		fmt.Printf("Retrieved chunks: %d\n", len(paragraphs))
		fmt.Printf("Precision@K: %.2f\n", precision)

		if !test.outOfScope {
			fmt.Printf("Recall@K: %.2f\n", recall)
			fmt.Printf("Missed relevant IDs: %v\n", missedIDs)
			fmt.Printf("Irrelevant IDs: %v\n", irrelevantIDs)

			metrics.precisionSum += precision
			metrics.recallSum += recall
			metrics.evaluated++
		} else {
			rejected := len(paragraphs) == 0

			fmt.Printf("Rejected: %t\n", rejected)

			metrics.outOfScope++
			if rejected {
				metrics.rejected++
			}
		}
	}

	fmt.Println("\n---------- Summary ----------")

	if metrics.evaluated > 0 {
		fmt.Printf(
			"Mean Precision@K: %.2f\n",
			metrics.precisionSum/float64(metrics.evaluated),
		)
		fmt.Printf(
			"Mean Recall@K: %.2f\n",
			metrics.recallSum/float64(metrics.evaluated),
		)
	}

	if metrics.outOfScope > 0 {
		fmt.Printf(
			"Out-of-scope rejection rate: %.2f\n",
			float64(metrics.rejected)/float64(metrics.outOfScope),
		)
	}

	return nil
}

func runEvaluation(topK int, minSimilarity float64) error {
	data, err := os.ReadFile("data/knowledge.txt")
	if err != nil {
		return fmt.Errorf("read evaluation corpus: %w", err)
	}

	paragraphs := splitParagraphs(string(data))

	expectedIDs := map[string]bool{
		"rag":               false,
		"embedding":         false,
		"retrieval":         false,
		"generation":        false,
		"vector_search":     false,
		"cosine_similarity": false,
		"chunking":          false,
		"top_k":             false,
		"threshold":         false,
		"hallucination":     false,
		"keyword_search":    false,
		"pgvector":          false,
	}

	if len(paragraphs) != len(expectedIDs) {
		return fmt.Errorf(
			"expected %d document chunks, got %d",
			len(expectedIDs),
			len(paragraphs),
		)
	}

	for _, paragraph := range paragraphs {
		id := identifyChunk(paragraph)

		if id == "unknown" {
			return fmt.Errorf(
				"unknown document chunk: %q",
				paragraph,
			)
		}

		if expectedIDs[id] {
			return fmt.Errorf(
				"duplicate document chunk: %s",
				id,
			)
		}

		expectedIDs[id] = true
	}

	for id, found := range expectedIDs {
		if !found {
			return fmt.Errorf(
				"missing document chunk: %s",
				id,
			)
		}
	}

	if err := evaluateMethod(
		"KEYWORD RETRIEVAL",
		func(path, query string, k int) (string, error) {
			return retrieve(path, query, k)
		},
		topK,
	); err != nil {
		return err
	}

	if err := evaluateMethod(
		"SEMANTIC RETRIEVAL",
		func(path, query string, k int) (string, error) {
			return semanticRetrieveWithThreshold(
				path,
				query,
				k,
				minSimilarity,
			)
		},
		topK,
	); err != nil {
		return err
	}

	return nil
}
