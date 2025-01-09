package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"math"
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
	Entries []KnowledgeEntry
	mu      sync.RWMutex
	LearnedEntries map[string]string
}

func (kb *KnowledgeBase) Learn(question, answer string) {
	kb.mu.Lock()
	defer kb.mu.Unlock()
	kb.LearnedEntries[question] = answer
}

type AIEngine struct {
	KB         *KnowledgeBase
	Embeddings map[string][]float64
}

func NewKnowledgeBase() *KnowledgeBase {
	return &KnowledgeBase{
		Entries: []KnowledgeEntry{},
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

func NewAIEngine(embeddings map[string][]float64) *AIEngine {
	kb := NewKnowledgeBase()
	kb.AddEntry("Что такое горутина в Go?", "Горутины — это легковесные потоки, управляемые рантаймом Go, позволяющие выполнять конкурентные задачи с минимальными ресурсами.", embeddings)
	kb.AddEntry("Что такое канал в Go?", "Каналы — это средства коммуникации между горутинами, позволяющие безопасно обмениваться данными согласно принципам CSP.", embeddings)
	kb.AddEntry("Что такое интерфейс в Go?", "Интерфейсы в Go определяют поведение через методы, обеспечивая полиморфизм и гибкость дизайна кода.", embeddings)
	return &AIEngine{
		KB:         kb,
		Embeddings: embeddings,
	}
}

func (ai *AIEngine) GenerateAnswer(question string) string {
	answer, score := ai.KB.FindBestMatch(question, ai.Embeddings)
	if score > 0.5 {
		return answer
	}
	doc, _ := prose.NewDocument(question)
	tokens := doc.Tokens()
	var keywords []string
	for _, tok := range tokens {
		if tok.Tag == "NOUN" || tok.Tag == "PROPN" {
			keywords = append(keywords, tok.Text)
		}
	}
	details := strings.Join(keywords[:min(3, len(keywords))], ", ")
	return fmt.Sprintf("Используя встроенные функции Go и учитывая такие концепции, как %s, можно решить этот вопрос более эффективно.", details)
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
	fmt.Println("Server starting on http://localhost:8080")
	http.ListenAndServe("localhost:8080", nil)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
