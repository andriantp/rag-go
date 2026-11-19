package main

import (
	"bufio"
	"os"
	"sort"
	"strings"
	"unicode"
)

type searchResult struct {
	text  string
	score int
}

var stopWords = map[string]struct{}{
	"a": {}, "an": {}, "the": {},
	"is": {}, "are": {}, "was": {}, "were": {},
	"what": {}, "which": {}, "who": {},
	"do": {}, "does": {}, "did": {},
	"how": {}, "why": {}, "when": {}, "where": {},
	"to": {}, "of": {}, "for": {}, "in": {},
	"on": {}, "at": {}, "by": {}, "with": {},
	"and": {}, "or": {}, "but": {},
	"it": {}, "its": {}, "this": {}, "that": {},
	"stand": {}, "stands": {},
}

func tokenize(text string) []string {
	fields := strings.FieldsFunc(
		strings.ToLower(text),
		func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		},
	)

	var tokens []string
	for _, token := range fields {
		if _, excluded := stopWords[token]; excluded {
			continue
		}
		if len([]rune(token)) < 2 {
			continue
		}
		tokens = append(tokens, token)
	}

	return tokens
}

func splitParagraphs(data string) []string {
	scanner := bufio.NewScanner(strings.NewReader(data))

	var paragraphs []string
	var lines []string

	flush := func() {
		if len(lines) == 0 {
			return
		}

		paragraphs = append(
			paragraphs,
			strings.Join(lines, " "),
		)
		lines = nil
	}

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		if line == "" {
			flush()
			continue
		}

		lines = append(lines, line)
	}

	flush()
	return paragraphs
}

func retrieve(path, query string, topK int) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}

	queryTokens := tokenize(query)
	if len(queryTokens) == 0 || topK <= 0 {
		return "", nil
	}

	querySet := make(map[string]struct{})
	for _, token := range queryTokens {
		querySet[token] = struct{}{}
	}

	paragraphs := splitParagraphs(string(data))
	var results []searchResult

	for _, paragraph := range paragraphs {
		paragraphTokens := tokenize(paragraph)

		paragraphSet := make(map[string]struct{})
		for _, token := range paragraphTokens {
			paragraphSet[token] = struct{}{}
		}

		score := 0
		for token := range querySet {
			if _, found := paragraphSet[token]; found {
				score++
			}
		}

		if score > 0 {
			results = append(results, searchResult{
				text:  paragraph,
				score: score,
			})
		}
	}

	sort.SliceStable(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if len(results) > topK {
		results = results[:topK]
	}

	var matches []string
	for _, result := range results {
		matches = append(matches, result.text)
	}

	return strings.Join(matches, "\n\n"), nil
}
