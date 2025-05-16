package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"github.com/pkoukk/tiktoken-go"
)

// GPTDatasetV1 holds the input and target token sequences
type GPTDatasetV1 struct {
	InputIDs  [][]int
	TargetIDs [][]int
}

// NewGPTDatasetV1 creates a new dataset from text
func NewGPTDatasetV1(txt string, tokenizer *tiktoken.Tiktoken, maxLength int, stride int) (*GPTDatasetV1, error) {
	dataset := &GPTDatasetV1{
		InputIDs:  make([][]int, 0),
		TargetIDs: make([][]int, 0),
	}

	// Tokenize the entire text
	tokenIDs := tokenizer.Encode(txt, nil, nil)

	// Use a sliding window to chunk the book into overlapping sequences of max_length
	for i := 0; i <= len(tokenIDs)-maxLength-1; i += stride {
		inputChunk := make([]int, maxLength)
		copy(inputChunk, tokenIDs[i:i+maxLength])

		targetChunk := make([]int, maxLength)
		copy(targetChunk, tokenIDs[i+1:i+maxLength+1])

		dataset.InputIDs = append(dataset.InputIDs, inputChunk)
		dataset.TargetIDs = append(dataset.TargetIDs, targetChunk)
	}
	return dataset, nil
}

// Len returns the number of samples in the dataset
func (d *GPTDatasetV1) Len() int {
	return len(d.InputIDs)
}

// GetItem returns the idx-th sample (input_ids, target_ids)
func (d *GPTDatasetV1) GetItem(idx int) ([]int, []int) {
	return d.InputIDs[idx], d.TargetIDs[idx]
}

// DataLoader represents a simplified data loader
type DataLoader struct {
	Dataset    *GPTDatasetV1
	BatchSize  int
	Shuffle    bool
	DropLast   bool
	Indices    []int
	CurrentIdx int
}

// NewDataLoader creates a new DataLoader
func NewDataLoader(dataset *GPTDatasetV1, batchSize int, shuffle bool, dropLast bool) *DataLoader {
	dl := &DataLoader{
		Dataset:    dataset,
		BatchSize:  batchSize,
		Shuffle:    shuffle,
		DropLast:   dropLast,
		Indices:    make([]int, dataset.Len()),
		CurrentIdx: 0,
	}

	for i := 0; i < dataset.Len(); i++ {
		dl.Indices[i] = i
	}

	if dl.Shuffle {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		r.Shuffle(len(dl.Indices), func(i, j int) {
			dl.Indices[i], dl.Indices[j] = dl.Indices[j], dl.Indices[i]
		})
	}
	return dl
}

// NextBatch returns the next batch of data.
func (dl *DataLoader) NextBatch() ([][]int, [][]int, bool) {
	if dl.CurrentIdx >= len(dl.Indices) {
		return nil, nil, false
	}

	endIdx := dl.CurrentIdx + dl.BatchSize
	if endIdx > len(dl.Indices) {
		if dl.DropLast || dl.CurrentIdx == len(dl.Indices) {
			return nil, nil, false
		}
		endIdx = len(dl.Indices)
	}
	if dl.CurrentIdx == endIdx {
		return nil, nil, false
	}

	batchInputIDs := make([][]int, 0, endIdx-dl.CurrentIdx)
	batchTargetIDs := make([][]int, 0, endIdx-dl.CurrentIdx)

	for i := dl.CurrentIdx; i < endIdx; i++ {
		inputIDs, targetIDs := dl.Dataset.GetItem(dl.Indices[i])
		batchInputIDs = append(batchInputIDs, inputIDs)
		batchTargetIDs = append(batchTargetIDs, targetIDs)
	}

	dl.CurrentIdx = endIdx
	return batchInputIDs, batchTargetIDs, true
}

// EmbeddingLayer represents a simple embedding layer
type EmbeddingLayer struct {
	Weights   [][]float64
	OutputDim int
}

// NewEmbeddingLayer creates a new embedding layer with random weights
func NewEmbeddingLayer(numEmbeddings int, outputDim int) *EmbeddingLayer {
	weights := make([][]float64, numEmbeddings)
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range weights {
		weights[i] = make([]float64, outputDim)
		for j := range weights[i] {
			weights[i][j] = r.NormFloat64()
		}
	}
	return &EmbeddingLayer{Weights: weights, OutputDim: outputDim}
}

// Forward performs the embedding lookup for a batch of IDs
func (el *EmbeddingLayer) Forward(batchIDs [][]int) [][][]float64 {
	batchSize := len(batchIDs)
	if batchSize == 0 {
		return [][][]float64{}
	}
	seqLength := len(batchIDs[0])

	batchEmbeddings := make([][][]float64, batchSize)
	for i := 0; i < batchSize; i++ {
		batchEmbeddings[i] = make([][]float64, seqLength)
		for j := 0; j < seqLength; j++ {
			tokenID := batchIDs[i][j]
			if tokenID >= 0 && tokenID < len(el.Weights) {
				// Create a copy of the embedding vector
				embeddingVector := make([]float64, el.OutputDim)
				copy(embeddingVector, el.Weights[tokenID])
				batchEmbeddings[i][j] = embeddingVector
			} else {
				batchEmbeddings[i][j] = make([]float64, el.OutputDim)
			}
		}
	}
	return batchEmbeddings
}

// ForwardSingle performs embedding lookup for a single sequence of IDs (e.g., arange)
func (el *EmbeddingLayer) ForwardSingle(ids []int) [][]float64 {
	seqLength := len(ids)
	embeddings := make([][]float64, seqLength)
	for i := 0; i < seqLength; i++ {
		id := ids[i]
		if id >= 0 && id < len(el.Weights) {
			embeddingVector := make([]float64, el.OutputDim)
			copy(embeddingVector, el.Weights[id])
			embeddings[i] = embeddingVector
		} else {
			embeddings[i] = make([]float64, el.OutputDim)
		}
	}
	return embeddings
}

func main() {
	vocabSize := 50257
	outputDim := 256
	contextLength := 1024
	batchSize := 8
	maxLength := 4
	stride := maxLength

	rawTextBytes, err := os.ReadFile("the-verdict.txt")
	if err != nil {
		fmt.Println("Warning: 'the-verdict.txt' not found.")
	}
	rawText := string(rawTextBytes)

	tke, err := tiktoken.GetEncoding("cl100k_base")
	if err != nil {
		log.Fatalf("Failed to get_encoding: %v", err)
	}

	dataset, err := NewGPTDatasetV1(rawText, tke, maxLength, stride)
	if err != nil {
		log.Fatalf("Failed to create dataset: %v", err)
	}
	if dataset.Len() == 0 {
		log.Fatalf("Dataset is empty. Ensure 'the-verdict.txt' has enough content and maxLength/stride are appropriate.")
	}

	dataloader := NewDataLoader(dataset, batchSize, true, true)
	tokenEmbeddingLayer := NewEmbeddingLayer(vocabSize, outputDim)
	posEmbeddingLayer := NewEmbeddingLayer(contextLength, outputDim)

	fmt.Println("Processing one batch...")
	xBatch, yBatch, hasNext := dataloader.NextBatch()

	if !hasNext || len(xBatch) == 0 {
		log.Fatalf("Dataloader returned no batches or an empty batch. Dataset length: %d, Batch size: %d", dataset.Len(), batchSize)
	}
	fmt.Printf("Batch X length: %d, Batch Y length: %d\n", len(xBatch), len(yBatch))
	if len(xBatch) > 0 {
		fmt.Printf("First item in X batch length: %d\n", len(xBatch[0]))
	}

	// Token Embeddings
	tokenEmbeddings := tokenEmbeddingLayer.Forward(xBatch)
	arangeMaxLength := make([]int, maxLength)
	for i := 0; i < maxLength; i++ {
		arangeMaxLength[i] = i
	}

	posEmbeddingsSingle := posEmbeddingLayer.ForwardSingle(arangeMaxLength)
	inputEmbeddings := make([][][]float64, len(tokenEmbeddings))
	for i := 0; i < len(tokenEmbeddings); i++ {
		inputEmbeddings[i] = make([][]float64, maxLength)
		for j := 0; j < maxLength; j++ {
			inputEmbeddings[i][j] = make([]float64, outputDim)
			if j < len(posEmbeddingsSingle) {
				for k := 0; k < outputDim; k++ {
					inputEmbeddings[i][j][k] = tokenEmbeddings[i][j][k] + posEmbeddingsSingle[j][k]
				}
			} else {
				copy(inputEmbeddings[i][j], tokenEmbeddings[i][j])
			}
		}
	}

	if len(inputEmbeddings) > 0 {
		if len(inputEmbeddings[0]) > 0 {
			fmt.Printf("Input Embeddings Shape: [%d, %d, %d]\n", len(inputEmbeddings), len(inputEmbeddings[0]), len(inputEmbeddings[0][0]))
		} else {
			fmt.Printf("Input Embeddings Shape: [%d, 0, 0] (Sequence length is zero)\n", len(inputEmbeddings))
		}
	} else {
		fmt.Println("Input Embeddings Shape: [0, 0, 0] (Batch size is zero)")
	}
	fmt.Println("Processing finished.")
}
