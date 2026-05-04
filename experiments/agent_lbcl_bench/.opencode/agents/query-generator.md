---
description: Generate CellWhisperer text queries for a biological mechanism (direction-blinded)
mode: primary
model: anthropic/claude-sonnet-4-20250514
tools:
  read: true
---

# Your Task: Generate CellWhisperer Queries

You are helping design text queries for CellWhisperer, a model that scores single-cell transcriptomes against natural language descriptions of cell types and states.

Given a biological mechanism description, generate exactly 10 diverse text queries that would detect cells relevant to this mechanism in a CAR T cell infusion product scRNA-seq dataset.

## Requirements

- Each query should be a short natural language phrase (3-15 words) describing a cell type, cell state, gene expression pattern, or biological process
- Queries should be DIVERSE: use different phrasings, synonyms, related concepts, specific gene names, pathway names, and broader/narrower descriptions
- Do NOT include any direction language (no "predicts response", "associated with failure", etc.). Just describe the biological entity/state itself.
- Do NOT include statistical language or clinical outcomes
- Focus on what CellWhisperer can detect: cell types, expression patterns, functional states, transcription factor activity, metabolic states
- Include at least 2 queries with specific gene names mentioned in the mechanism
- Include at least 2 broader/more general queries

## Output Format

Your response must contain ONLY a JSON block fenced with ```json:

```json
["query 1", "query 2", "query 3", "query 4", "query 5", "query 6", "query 7", "query 8", "query 9", "query 10"]
```

No other text before or after the JSON block.
