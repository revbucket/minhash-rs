# MinHash Debug Scripts - Generated Notes

Python utilities for analyzing and debugging MinHash LSH results.

## Quick Start

### 1. Run MinHash and Component Analysis

```bash
# Step 1: Run MinHash LSH
cargo run --release -- minhash-memory \
  --input-dir /path/to/documents \
  --storage-dir /path/to/storage \
  --output-dir /path/to/output \
  --text-key "text" \
  --config config.yaml

# Step 2: Run component analysis
cargo run --release -- component-analysis \
  --input-dir /path/to/documents \
  --storage-dir /path/to/storage \
  --output-file component_analysis.jsonl \
  --text-key "text" \
  --config config.yaml
```

### 2. Analyze Results

```bash
# Get overall statistics
python3 analyze_component_distributions.py component_analysis.jsonl

# View a specific component
python3 view_component.py component_analysis.jsonl /path/to/documents <cc_id>

# Interactive side-by-side viewer (requires similarities file)
python3 view_similarities.py similarities.jsonl
```

---

## Configuration

MinHash behavior is controlled via a YAML config file. Key parameters:

### MinHash Parameters

```yaml
minhash_params:
  num_buckets: 26        # B - Number of bands
  bucket_size: 11        # R - Rows per band
  ngram_size: 5          # N-gram size for shingling
  permutation_seed: 42   # Random seed for reproducibility
  tokenizer: "cl100k_base"  # Tokenizer (cl100k_base or whitespace)

eng_params:
  num_docs: 6000              # Expected number of documents
  max_lines_per_path: 6000    # Max lines per input file
  num_sig_chunks: 16          # Parallel processing chunks

output_params:
  annotate: true               # Add annotations to output
  annotate_key: "true_jaccard" # Key for annotation field
  remove_duplicates: false     # Remove duplicates from output
  delete_while_cleaning: false # Delete files during cleaning
```

### Tuning B and R for Different Thresholds

The trade-off between bands (B) and rows (R) determines your similarity threshold:

| Target Similarity | B (bands) | R (rows) | Precision | Recall |
|-------------------|-----------|----------|-----------|--------|
| 70-80% (broad)    | 26        | 11       | Medium    | High   |
| 80-85% (balanced) | 19        | 15       | High      | High   |
| 90%+ (strict)     | 11        | 25       | Very High | Medium |

**Rule of thumb:**
- **More bands (B ↑)**: Higher recall, catches more pairs (but more false positives)
- **More rows (R ↑)**: Higher precision, fewer false positives (but misses some true pairs)
- **Inflection point**: s ≈ (1/B)^(1/R)

---

## Scripts Reference

### analyze_component_distributions.py

**Purpose:** Statistical analysis of MinHash component quality

**What it does:**
- Computes Jaccard similarity distributions for direct edges (MinHash matches) and transitive edges (Union-Find only)
- Shows histograms of similarity scores
- Compares precision/recall characteristics
- Identifies potential false positives (low-Jaccard direct edges)

**Usage:**
```bash
python3 analyze_component_distributions.py <component_analysis.jsonl>
```

**Example output:**
```
======================================================================
COMPONENT ANALYSIS SUMMARY
======================================================================
Total Components:           1456
Fully Connected:            1286 (88.3%)
With Transitive Edges:      170 (11.7%)
Total Direct Edges:         8146
Total Transitive Edges:     16201
Direct Edge Rate:           33.5%

DIRECT EDGES (MinHash Band Matches)
Count: 8146
Mean:  0.7916
Min:   0.2404
Max:   1.0000
...
```

**Use this to:**
- Validate your B/R settings
- Check for excessive false positives (low-Jaccard direct edges)
- Understand component connectivity

---

### view_component.py

**Purpose:** Deep dive into a specific component

**What it does:**
- Shows all documents in a component with text previews
- Lists all direct edges with Jaccard scores and band match counts
- Lists all transitive edges (pairs connected via Union-Find but no direct band match)
- Displays component-level statistics

**Usage:**
```bash
python3 view_component.py <component_analysis.jsonl> <input_dir> <cc_id>
```

**Example:**
```bash
# View component 11857
python3 view_component.py component_analysis.jsonl /path/to/docs 11857
```

**Example output:**
```
################################################################################
# Component 11857 - Size: 6
# Direct Edges: 15 | Transitive Edges: 0
# Density: 100.00% | Fully Connected: True
################################################################################

Direct Edge Statistics:
  Mean Jaccard: 0.4467
  Min:  0.2404
  Max:  1.0000

================================================================================
DOCUMENTS IN COMPONENT
================================================================================

================================================================================
Document: doc_9256
================================================================================
Systems and methods are provided for presenting subtitles in association with...

================================================================================
DIRECT EDGES (MinHash Band Matches)
================================================================================
doc_9256 <-> doc_10382: Jaccard=0.2404, Bands=[0]
doc_9256 <-> doc_13409: Jaccard=0.2447, Bands=[0]
...
```

**Use this to:**
- Investigate specific components (especially those with low-Jaccard matches)
- Understand why documents were grouped together
- Debug false positives or false negatives

**Finding interesting components:**
```bash
# Find components with low-Jaccard direct edges
cat component_analysis.jsonl | python3 -c "
import json
import sys
for line in sys.stdin:
    comp = json.loads(line)
    for edge in comp.get('direct_edges', []):
        if edge['jaccard'] < 0.4:
            print(f\"CC {comp['cc_id']}: {edge['doc1_id']} <-> {edge['doc2_id']} = {edge['jaccard']:.4f}\")
" | head -10
```

---

### view_similarities.py

**Purpose:** Interactive side-by-side document comparison

**What it does:**
- Displays two documents side-by-side with START and END sections
- Shows Jaccard similarity score
- Interactive navigation (next/previous/jump to index)
- Sorted by similarity score (highest first)

**Usage:**
```bash
python3 view_similarities.py <similarities.jsonl>
```

**Input format:**
The similarities file must contain records with these fields:
```json
{
  "doc1_id": "doc_123",
  "doc2_id": "doc_456",
  "jaccard_score": 0.85,
  "doc1_text_start": "First 500 chars of doc 1...",
  "doc1_text_end": "Last 500 chars of doc 1...",
  "doc2_text_start": "First 500 chars of doc 2...",
  "doc2_text_end": "Last 500 chars of doc 2..."
}
```

**Interactive controls:**
- `n` - Next pair
- `p` - Previous pair
- `j` - Jump to specific index
- `q` - Quit

**Use this to:**
- Visually compare similar documents
- Identify common patterns (e.g., boilerplate text)
- Validate similarity scores

**Note:** Component analysis output doesn't include text snippets by default. You'll need to extract specific pairs and add the text fields manually, or use a tool that generates this format.

---

## Common Workflows

### Workflow 1: Validate MinHash Settings

```bash
# 1. Run MinHash with your current settings
cargo run --release -- minhash-memory --config config.yaml ...

# 2. Run component analysis
cargo run --release -- component-analysis ...

# 3. Check overall quality
python3 analyze_component_distributions.py component_analysis.jsonl

# 4. Look for issues
# - If mean Jaccard < 0.7: Too many false positives → increase R
# - If very few edges: Too strict → decrease R or increase B
```

### Workflow 2: Investigate False Positives

```bash
# 1. Find components with low-Jaccard edges
cat component_analysis.jsonl | python3 -c "
import json
import sys
for line in sys.stdin:
    comp = json.loads(line)
    if comp['stats']['direct_jaccard_min'] < 0.3:
        print(f\"Component {comp['cc_id']}: min={comp['stats']['direct_jaccard_min']:.4f}, size={comp['cc_size']}\")
"

# 2. Examine specific component
python3 view_component.py component_analysis.jsonl /path/to/docs <cc_id>

# 3. Look at the documents to understand why they matched
# - Check for boilerplate text
# - Check number of band matches (low count suggests chance collision)
```

### Workflow 3: Tune for Your Dataset

```bash
# Test different (B, R) combinations
for config in config_26x11.yaml config_19x15.yaml config_11x25.yaml; do
    echo "Testing $config..."

    # Run MinHash
    cargo run --release -- minhash-memory --config $config \
        --input-dir input --storage-dir storage_$config --output-dir output_$config

    # Analyze
    cargo run --release -- component-analysis \
        --storage-dir storage_$config --output-file results_$config.jsonl

    # Get stats
    python3 analyze_component_distributions.py results_$config.jsonl > stats_$config.txt
done

# Compare results to find optimal settings
```

---

## Understanding the Output

### Component Types

1. **Fully Connected Components**
   - All pairs matched directly via MinHash bands
   - Density = 100%
   - Usually high-quality clusters

2. **Components with Transitive Edges**
   - Some pairs connected only via Union-Find
   - Indicates partial matches or star-shaped clusters
   - Transitive edges typically have lower Jaccard than direct edges

### Band Match Counts

When viewing components, the `Bands=[...]` field shows which bands matched:

```
doc_123 <-> doc_456: Jaccard=0.95, Bands=[0, 0, 0, 1, 1, 1, 2, ...]
```

- More band matches → stronger signal, less likely to be false positive
- Single band match + low Jaccard → likely false positive from boilerplate
- All bands match → documents are identical or near-identical

---

## Troubleshooting

### Issue: Too many low-Jaccard matches

**Symptoms:** `analyze_component_distributions.py` shows many direct edges with Jaccard < 0.5

**Solutions:**
1. Increase R (rows per band) to make matching stricter
2. Try B=19, R=15 or B=11, R=25

### Issue: Very few matches found

**Symptoms:** Very few components, most documents isolated

**Solutions:**
1. Decrease R or increase B to be more permissive
2. Check your documents are formatted correctly (JSONL with text field)
3. Verify tokenizer setting matches your data

### Issue: Component analysis is slow

**Solutions:**
1. Component analysis computes true Jaccard for all pairs - this is O(n²) per component
2. For large components, consider sampling or splitting into smaller batches
3. Increase `num_sig_chunks` in config for more parallelism

---

## Additional Resources

- MinHash LSH theory: https://web.stanford.edu/class/cs246/slides/03-lsh.pdf
- S-curve analysis: https://ekzhu.com/datasketch/lsh.html
- Patent data characteristics: Patent text often has significant boilerplate, so higher R values (15-25) work better than general web text

