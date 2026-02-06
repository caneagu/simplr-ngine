# LLM Prompts Documentation

This document documents all LLM prompts used in the Simplr system and provides guidance on optimizing them for better RAG accuracy and response quality.

## Current Prompt Locations

All prompts are hardcoded in `app/services/llm.py`. There are currently **4 main prompts**:

1. **Summarization Prompt** (`summarize_text()`)
2. **Classification Prompt** (`categorize_and_extract()`)
3. **Q&A Prompt** (`answer_with_context()`)
4. **Streaming Q&A Prompt** (`stream_answer_with_context()`)

## Current Prompts

### 1. Summarization Prompt

**Function**: `summarize_text()`
**Purpose**: Generate a bullet-point summary of email/article content
**Model**: Configurable via `LLM_MODEL` env var (default: gpt-4o-mini)

```python
system_prompt = """
You summarize incoming knowledge into a concise knowledge-base article summary.
"""

user_prompt = """
Summarize the following content in 4-7 bullet points.

{content}
"""
```

**Output Format**: Plain text, bullet points
**Temperature**: 0.2 (deterministic)

### 2. Classification & Extraction Prompt

**Function**: `categorize_and_extract()`
**Purpose**: Classify content into categories and extract metadata/entities
**Model**: Configurable via `LLM_MODEL` env var

```python
system_prompt = """
You are a classifier. Return JSON only with keys: category, confidence, metadata, entities, version.
"""

user_prompt = """
Classify the content into one of: support_tickets, policies, documentation, projects, other.
Extract metadata fields if present: people (array), dates (array), timeline (string), progress (string).
Also extract entities with keys: people (array), orgs (array), locations (array), tags (array).
Set confidence from 0 to 1.
Return JSON only.

Content:
{content}
"""
```

**Output Format**: JSON
```json
{
  "category": "projects",
  "confidence": 0.92,
  "metadata": {
    "people": ["Alice", "Bob"],
    "dates": ["2024-01-15"],
    "timeline": "Q1 2024",
    "progress": "75% complete"
  },
  "entities": {
    "people": ["Alice", "Bob", "Charlie"],
    "orgs": ["Acme Corp"],
    "locations": ["New York"],
    "tags": ["urgent", "review"]
  },
  "version": 1
}
```

### 3. Q&A Prompt

**Function**: `answer_with_context()`
**Purpose**: Answer user questions using retrieved context
**Model**: Configurable via `LLM_MODEL` env var

```python
system_prompt = """
You answer questions using the provided knowledge base context.
If the answer is not in the context, say you don't know.
"""

user_prompt = """
Context:
{context}

Question:
{question}

Answer in a concise, helpful way.
"""
```

### 4. Streaming Q&A Prompt

**Function**: `stream_answer_with_context()`
**Purpose**: Stream token-by-token responses with conversation history
**Model**: Configurable via `LLM_MODEL` env var

```python
system_prompt_1 = """
You answer questions using the provided knowledge base context.
If the answer is not in the context, say you don't know.
"""

system_prompt_2 = """
Conversation history:
{history}
"""

user_prompt = """
Context:
{context}

Question:
{question}

Answer in a concise, helpful way.
"""
```

## Prompt Engineering Best Practices for RAG

### 1. Context Grounding

**Current**: Basic grounding with "If the answer is not in the context, say you don't know."

**Improved Approach**:
```python
system_prompt = """
You are a helpful assistant that answers questions based ONLY on the provided knowledge base context.

CRITICAL RULES:
1. Base your answer EXCLUSIVELY on the provided context
2. If the context doesn't contain the answer, respond: "I don't have enough information to answer that question."
3. Do not use outside knowledge or make assumptions
4. If the context is partial, indicate what information is missing
5. Cite specific sources from the context when possible

When uncertain, ask clarifying questions rather than guessing.
"""
```

### 2. Context Formatting

**Current**: Simple concatenation

**Improved Approach** with source tracking:
```python
def _build_context(chunks: list[tuple[Chunk, Article]]) -> str:
    """Build context with source attribution for better citation."""
    parts = []
    for i, (chunk, article) in enumerate(chunks, 1):
        parts.append(
            f"[SOURCE {i}]\n"
            f"Article: {article.title}\n"
            f"Category: {article.metadata_.get('category', 'uncategorized')}\n"
            f"Content: {chunk.content}\n"
        )
    return "\n\n".join(parts)
```

### 3. Prompt Templates for Better Control

**Recommended Template Structure**:

```python
RAG_PROMPT_TEMPLATE = """You are a precise knowledge base assistant. Answer the user's question using ONLY the information provided in the context below.

<CONTEXT>
{context}
</CONTEXT>

<INSTRUCTIONS>
1. Read the context carefully
2. Extract only information relevant to the question
3. If multiple sources provide information, synthesize them
4. If the answer isn't in the context, say "I don't have enough information"
5. Keep responses concise (2-4 sentences ideally)
6. Be factual and avoid speculation
</INSTRUCTIONS>

<QUESTION>
{question}
</QUESTION>

Answer:"""
```

### 4. Few-Shot Examples

Add examples to improve consistency:

```python
system_prompt = """
You answer questions using the provided knowledge base context.

EXAMPLES:

Context: "Q1 revenue was $1.2M, up 25% from Q4."
Question: "What was Q1 revenue?"
Answer: "Q1 revenue was $1.2 million."

Context: "The project deadline is March 15, 2024."
Question: "When is the deadline?"
Answer: "The deadline is March 15, 2024."

Context: "Alice is leading the engineering team."
Question: "Who is the CEO?"
Answer: "I don't have enough information to answer that question."

Now answer based on the context provided.
"""
```

## Optimization Strategies

### 1. Reduce Hallucinations

**Technique**: Explicit constraint reinforcement

```python
system_prompt = """
You are a knowledge base assistant. Follow these rules STRICTLY:

✓ DO:
  - Use ONLY information from the provided context
  - Say "I don't know" if information is missing
  - Quote relevant text when appropriate
  - Be concise and direct

✗ DON'T:
  - Make up information not in the context
  - Use general knowledge outside the context
  - Speculate or assume
  - Add disclaimers about being an AI

Your job is to extract and present information accurately.
"""
```

### 2. Improve Relevance

**Technique**: Query rewriting for better context retrieval

```python
def rewrite_query_for_retrieval(question: str) -> str:
    """Expand query to improve semantic search recall."""
    # This could be a separate LLM call or rules-based
    prompt = f"""Given this user question, rewrite it to be more specific and detailed for searching a knowledge base.
    
Original: {question}

Rewritten (be specific about what information is needed):"""
    # Return expanded query for embedding
```

### 3. Handle Edge Cases

```python
system_prompt = """
You answer questions using the provided knowledge base context.

SPECIAL CASES:

1. EMPTY CONTEXT: If no context is provided, say "I don't have any relevant information about that."

2. CONFLICTING INFO: If sources conflict, present both perspectives and note the discrepancy.

3. OUTDATED INFO: If dates are present, consider if information might be outdated.

4. UNCLEAR QUESTION: If the question is ambiguous, ask for clarification.

5. MULTIPLE ANSWERS: If the context contains multiple relevant facts, synthesize them.

Always prioritize accuracy over completeness.
"""
```

### 4. Chain-of-Thought for Complex Questions

```python
system_prompt = """
You answer questions using the provided knowledge base context.

For complex questions, think step by step:

1. IDENTIFY: What is the user asking for?
2. LOCATE: Which parts of the context are relevant?
3. EXTRACT: What specific information answers the question?
4. VERIFY: Is the answer fully supported by the context?
5. RESPOND: Provide the answer or state if information is missing

Keep the reasoning internal and provide only the final answer.
"""
```

## Temperature Settings

| Use Case | Temperature | Reasoning |
|----------|-------------|-----------|
| **Summarization** | 0.2 | Consistent, factual output |
| **Classification** | 0.1 | Deterministic category selection |
| **Q&A** | 0.2 | Factual answers, low creativity |
| **Entity Extraction** | 0.1 | Precise extraction |
| **Brainstorming** | 0.7 | More creative outputs |

## Recommended Prompt Improvements

### Summary: Current vs Recommended

| Aspect | Current | Recommended |
|--------|---------|-------------|
| **Grounding** | Weak instruction | Explicit "use only context" rules |
| **Format** | Simple text | Structured with XML tags |
| **Examples** | None | Include few-shot examples |
| **Edge Cases** | Not handled | Specific handling rules |
| **Citations** | None | Source attribution |
| **Length Control** | None | "concise" instruction |

### Implementation Priority

1. **High Priority** (Immediate impact):
   - Strengthen grounding instructions
   - Add source attribution to context
   - Include few-shot examples

2. **Medium Priority** (Quality improvements):
   - Implement query rewriting
   - Add chain-of-thought for complex queries
   - Fine-tune temperature per use case

3. **Low Priority** (Nice to have):
   - Add confidence scores
   - Implement response formatting templates
   - A/B test prompt variations

## Testing Prompt Changes

### Evaluation Framework

```python
# tests/test_prompts.py
test_cases = [
    {
        "name": "Direct answer from context",
        "context": "Revenue was $1M in Q1.",
        "question": "What was Q1 revenue?",
        "expected": "Should answer '$1M'",
        "must_not_contain": ["I don't know", "not mentioned"]
    },
    {
        "name": "Answer not in context",
        "context": "Revenue was $1M in Q1.",
        "question": "What was Q2 revenue?",
        "expected": "Should say 'I don't know'",
        "must_not_contain": ["$", "million", "revenue was"]
    },
    {
        "name": "Hallucination test",
        "context": "Alice is the manager.",
        "question": "What is Alice's email?",
        "expected": "Should say information not available",
        "must_not_contain": ["@", ".com", "email is"]
    }
]
```

### Metrics to Track

1. **Accuracy**: % of questions answered correctly from context
2. **Hallucination Rate**: % of answers containing invented information
3. **Refusal Rate**: % of times correctly saying "I don't know"
4. **Relevance Score**: Human rating of answer relevance (1-5)
5. **Response Length**: Average token count (shorter is usually better for RAG)

## Future Prompt Management

### Option 1: External Configuration File

```python
# prompts.yaml
summarization:
  system: "You summarize..."
  temperature: 0.2
  
classification:
  system: "You are a classifier..."
  output_schema:
    category: str
    confidence: float
```

### Option 2: Database-Driven Prompts

```python
# Allow prompts to be updated without redeployment
class Prompt(Base):
    __tablename__ = "prompts"
    id = Column(UUID, primary_key=True)
    name = Column(String, unique=True)
    system_prompt = Column(Text)
    user_template = Column(Text)
    temperature = Column(Float)
    version = Column(Integer)
```

### Option 3: Prompt Versioning

```python
# Version prompts for A/B testing and rollback
PROMPT_VERSIONS = {
    "qa_v1": {...},  # Current
    "qa_v2": {...},  # Experimental
}

# Use feature flags to control rollout
if feature_flags.enabled("prompt_v2"):
    prompt = PROMPT_VERSIONS["qa_v2"]
```

## Quick Wins for Better RAG

1. **Add source citations** to context building
2. **Strengthen grounding** with explicit constraints
3. **Lower temperature** to 0.1-0.2 for all RAG operations
4. **Add examples** to the system prompt
5. **Include instruction** to be concise
6. **Test edge cases** specifically
7. **Monitor** hallucination rate in production

## References

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/)
- [Anthropic Claude Prompt Design](https://docs.anthropic.com/claude/docs/prompt-design)
- [Microsoft RAG Survey](https://arxiv.org/abs/2312.10997)
