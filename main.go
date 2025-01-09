package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"net/http"
	"strings"
	"sync"

	"github.com/jdkato/prose/v2"
)

type AIResponse struct {
	Answer string `json:"answer"`
}

type Question struct {
	Text string `json:"text"`
}

type KnowledgeEntry struct {
	Question string
	Answer   string
	Vector   []float64
}

type KnowledgeBase struct {
	Entries        []KnowledgeEntry
	mu             sync.RWMutex
	LearnedEntries map[string]string
}

func (kb *KnowledgeBase) Learn(question, answer string) {
	kb.mu.Lock()
	defer kb.mu.Unlock()
	kb.LearnedEntries[question] = answer
}

type AIEngine struct {
	KB               *KnowledgeBase
	Embeddings       map[string][]float64
	Greetings        map[string]string
	CommonQuestions  map[string]string
	DefaultResponses map[string]string
	ContextMemory    []Interaction
	Patterns         map[string]float64
}

type Interaction struct {
	Question string
	Answer   string
	Keywords []string
	Score    float64
}

func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		Entries:        []KnowledgeEntry{},
		LearnedEntries: make(map[string]string),
	}
}

func (kb *KnowledgeBase) AddEntry(question, answer string, embeddings map[string][]float64) {
	vector := getSentenceVector(question, embeddings)
	kb.mu.Lock()
	kb.Entries = append(kb.Entries, KnowledgeEntry{
		Question: question,
		Answer:   answer,
		Vector:   vector,
	})
	kb.mu.Unlock()
}

func (kb *KnowledgeBase) FindBestMatch(question string, embeddings map[string][]float64) (string, float64) {
	queryVec := getSentenceVector(question, embeddings)
	kb.mu.RLock()
	defer kb.mu.RUnlock()
	var bestScore float64
	var bestAnswer string
	for _, entry := range kb.Entries {
		score := cosineSimilarity(queryVec, entry.Vector)
		if score > bestScore {
			bestScore = score
			bestAnswer = entry.Answer
		}
	}
	return bestAnswer, bestScore
}

func loadPrompts() (map[string]string, map[string]string, []KnowledgeEntry, map[string]string) {
	data, err := ioutil.ReadFile("prompt.json")
	if err != nil {
		log.Fatal("Error loading prompt.json:", err)
	}

	var config struct {
		Greetings       map[string]string `json:"greetings"`
		CommonQuestions map[string]string `json:"common_questions"`
		KnowledgeBase   []struct {
			Question string `json:"question"`
			Answer   string `json:"answer"`
		} `json:"knowledge_base"`
		DefaultResponses map[string]string `json:"default_responses"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		log.Fatal("Error parsing prompt.json:", err)
	}

	entries := make([]KnowledgeEntry, len(config.KnowledgeBase))
	for i, kb := range config.KnowledgeBase {
		entries[i] = KnowledgeEntry{
			Question: kb.Question,
			Answer:   kb.Answer,
		}
	}
	return config.Greetings, config.CommonQuestions, entries, config.DefaultResponses
}

func NewAIEngine(embeddings map[string][]float64) *AIEngine {
	kb := NewKnowledgeBase()
	greetings, commonQuestions, knowledgeBase, defaultResponses := loadPrompts()

	for _, entry := range knowledgeBase {
		kb.AddEntry(entry.Question, entry.Answer, embeddings)
	}

	return &AIEngine{
		KB:               kb,
		Embeddings:       embeddings,
		Greetings:        greetings,
		CommonQuestions:  commonQuestions,
		DefaultResponses: defaultResponses,
		Patterns:         make(map[string]float64),
	}
}

func (ai *AIEngine) findSimilarInteraction(keywords []string) (Interaction, float64) {
	var bestMatch Interaction
	var bestScore float64

	for _, interaction := range ai.ContextMemory {
		var matchCount int
		for _, k1 := range keywords {
			for _, k2 := range interaction.Keywords {
				if strings.ToLower(k1) == strings.ToLower(k2) {
					matchCount++
				}
			}
		}
		if len(keywords) > 0 {
			score := float64(matchCount) / float64(len(keywords))
			if score > bestScore {
				bestScore = score
				bestMatch = interaction
			}
		}
	}
	return bestMatch, bestScore
}

func (ai *AIEngine) GenerateAnswer(question string) string {
	keywords, concepts := ai.analyzeInput(question)
	contextScore := ai.evaluateContext(keywords)

	bestMatch, score := ai.findSimilarInteraction(keywords)
	if score > 0.8 {
		return ai.adaptResponse(bestMatch.Answer, keywords)
	}

	if answer, exists := ai.KB.LearnedEntries[question]; exists {
		adapted := ai.adaptResponse(answer, keywords)
		ai.learnFromInteraction(question, adapted, keywords, contextScore)
		return adapted
	}

	questionLower := strings.ToLower(question)

	if response, exists := ai.Greetings[questionLower]; exists {
		return response
	}

	for key, value := range ai.CommonQuestions {
		if strings.Contains(questionLower, key) {
			return value
		}
	}

	answer, score := ai.KB.FindBestMatch(question, ai.Embeddings)
	if score > 0.7 {
		return answer
	}

	doc, err := prose.NewDocument(question)
	if err != nil {
		return ai.DefaultResponses["error"]
	}

	keywords, concepts = ai.analyzeInput(question)

	if len(keywords) > 0 {
		techTerms := strings.Join(keywords[:min(3, len(keywords))], ", ")
		if defaultResponse, ok := ai.DefaultResponses["keywords"]; ok {
			return fmt.Sprintf(defaultResponse, techTerms)
		}
		return fmt.Sprintf("Let's explore %s in detail. What specific aspects interest you?", techTerms)
	}

	if defaultResponse, ok := ai.DefaultResponses["default"]; ok {
		return defaultResponse
	}

	starters := []string{
		"I'm here to help with Go programming. Could you specify what you'd like to learn about?",
		"I can assist you with various Go topics. What interests you most?",
		"Let me help you with Go! What would you like to explore?",
	}

	return starters[rand.Intn(len(starters))]
}

func (ai *AIEngine) analyzeInput(input string) ([]string, []string) {
	doc, err := prose.NewDocument(input)
	if err != nil {
		return nil, nil
	}

	var keywords, concepts []string
	for _, tok := range doc.Tokens() {
		switch tok.Tag {
		case "NOUN", "PROPN":
			keywords = append(keywords, tok.Text)
		case "VERB":
			concepts = append(concepts, tok.Text)
		}
	}
	return keywords, concepts
}

func (ai *AIEngine) evaluateContext(keywords []string) float64 {
	var score float64
	for _, word := range keywords {
		if weight, exists := ai.Patterns[word]; exists {
			score += weight
		}
	}
	return score / float64(len(keywords))
}

func (ai *AIEngine) adaptResponse(base string, keywords []string) string {
	if len(keywords) > 0 {
		return fmt.Sprintf("Based on %s, I understand that %s",
			strings.Join(keywords, ", "), base)
	}
	return base
}

func (ai *AIEngine) learnFromInteraction(q, a string, k []string, score float64) {
	interaction := Interaction{
		Question: q,
		Answer:   a,
		Keywords: k,
		Score:    score,
	}
	ai.ContextMemory = append(ai.ContextMemory, interaction)

	for _, keyword := range k {
		ai.Patterns[keyword] += 0.1 * score
	}
}

func cosineSimilarity(vec1, vec2 []float64) float64 {
	var dot, normA, normB float64
	for i := 0; i < len(vec1) && i < len(vec2); i++ {
		dot += vec1[i] * vec2[i]
		normA += vec1[i] * vec1[i]
		normB += vec2[i] * vec2[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func getSentenceVector(sentence string, embeddings map[string][]float64) []float64 {
	words := strings.Fields(strings.ToLower(sentence))
	var vec []float64
	for _, word := range words {
		if v, ok := embeddings[word]; ok {
			vec = addVectors(vec, v)
		}
	}
	return averageVector(vec, len(words))
}

func addVectors(a, b []float64) []float64 {
	if len(a) == 0 {
		return append([]float64{}, b...)
	}
	for i := 0; i < len(a) && i < len(b); i++ {
		a[i] += b[i]
	}
	return a
}

func averageVector(vec []float64, count int) []float64 {
	if count == 0 {
		return vec
	}
	for i := 0; i < len(vec); i++ {
		vec[i] /= float64(count)
	}
	return vec
}

func loadEmbeddings() map[string][]float64 {
	data, _ := ioutil.ReadFile("embeddings.json")
	var embeddings map[string][]float64
	json.Unmarshal(data, &embeddings)
	return embeddings
}

func handleAI(ai *AIEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}
		var question Question
		if err := json.NewDecoder(r.Body).Decode(&question); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		answer := ai.GenerateAnswer(question.Text)
		response := AIResponse{Answer: answer}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(response)
	}
}

func handleTemplates(w http.ResponseWriter, r *http.Request) {
	tmpl, _ := template.ParseGlob("templates/*")
	data := struct {
		Title string
	}{
		Title: "Go AI Assistant",
	}
	tmpl.ExecuteTemplate(w, "index.html", data)
}

type LearnRequest struct {
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

func handleLearn(ai *AIEngine) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Only POST method is allowed", http.StatusMethodNotAllowed)
			return
		}
		var req LearnRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		ai.KB.Learn(req.Question, req.Answer)
		w.WriteHeader(http.StatusOK)
	}
}

func main() {
	embeddings := loadEmbeddings()
	ai := NewAIEngine(embeddings)
	http.HandleFunc("/learn", handleLearn(ai))
	http.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.Dir("static"))))
	http.HandleFunc("/ai", handleAI(ai))
	http.HandleFunc("/", handleTemplates)
	fmt.Println("Server starting on http://0.0.0.0:8080")
	http.ListenAndServe("0.0.0.0:8080", nil)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
