#!/usr/bin/env python3
"""View documents in a MinHash component with similarity analysis."""

import json
import sys
from pathlib import Path

def load_component(component_file, cc_id):
    """Load a specific component from the analysis results."""
    with open(component_file) as f:
        for line in f:
            comp = json.loads(line)
            if comp['cc_id'] == cc_id:
                return comp
    return None

def load_documents(input_dir):
    """Load all documents from input directory."""
    docs = {}
    for jsonl_file in Path(input_dir).glob("*.jsonl"):
        with open(jsonl_file) as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                docs[f"{jsonl_file.stem}_{idx}"] = doc
    return docs

def extract_doc_id_number(doc_id):
    """Extract numeric ID from doc_XXX format."""
    return int(doc_id.split('_')[1])

def find_document_by_index(input_dir, doc_idx):
    """Find document by its MinHash index (not by JSON id field)."""
    # For the sampled dataset, there's only one file, so we can just index into it
    jsonl_files = list(Path(input_dir).glob("*.jsonl"))
    if not jsonl_files:
        return None

    # Read the file and get the doc at index doc_idx
    with open(jsonl_files[0]) as f:
        for idx, line in enumerate(f):
            if idx == doc_idx:
                return json.loads(line)
    return None

def print_document(doc_id, doc, max_len=500):
    """Print a document with truncation."""
    text = doc.get('text', 'NO TEXT')
    if len(text) > max_len:
        text = text[:max_len] + "..."
    print(f"\n{'='*80}")
    print(f"Document: {doc_id}")
    print(f"{'='*80}")
    print(text)

def visualize_component(component, input_dir):
    """Visualize a component with its documents and edges."""
    cc_id = component['cc_id']
    cc_size = component['cc_size']

    print(f"\n{'#'*80}")
    print(f"# Component {cc_id} - Size: {cc_size}")
    print(f"# Direct Edges: {component['num_direct_edges']} | Transitive Edges: {component['num_transitive_edges']}")
    print(f"# Density: {component['density']:.2%} | Fully Connected: {component['is_fully_connected']}")
    print(f"{'#'*80}")

    # Print stats
    stats = component['stats']
    print(f"\nDirect Edge Statistics:")
    print(f"  Mean Jaccard: {stats['direct_jaccard_mean']:.4f}")
    print(f"  Min:  {stats['direct_jaccard_min']:.4f}")
    print(f"  Max:  {stats['direct_jaccard_max']:.4f}")

    if component['num_transitive_edges'] > 0:
        print(f"\nTransitive Edge Statistics:")
        print(f"  Mean Jaccard: {stats.get('transitive_jaccard_mean', 0):.4f}")
        print(f"  Min:  {stats.get('transitive_jaccard_min', 0):.4f}")
        print(f"  Max:  {stats.get('transitive_jaccard_max', 0):.4f}")

    # Load documents
    print(f"\n{'='*80}")
    print("DOCUMENTS IN COMPONENT")
    print(f"{'='*80}")

    doc_ids = set()
    for edge in component['direct_edges'] + component['transitive_edges']:
        doc_ids.add(edge['doc1_id'])
        doc_ids.add(edge['doc2_id'])

    documents = {}
    for doc_id in sorted(doc_ids):
        doc_idx = extract_doc_id_number(doc_id)
        doc = find_document_by_index(input_dir, doc_idx)
        if doc:
            documents[doc_id] = doc
            print_document(doc_id, doc, max_len=400)

    # Print edges
    print(f"\n{'='*80}")
    print("DIRECT EDGES (MinHash Band Matches)")
    print(f"{'='*80}")
    for edge in sorted(component['direct_edges'], key=lambda e: e['jaccard'], reverse=True):
        bands = edge.get('bands', [])
        print(f"{edge['doc1_id']} <-> {edge['doc2_id']}: Jaccard={edge['jaccard']:.4f}, Bands={bands}")

    if component['num_transitive_edges'] > 0:
        print(f"\n{'='*80}")
        print("TRANSITIVE EDGES (Union-Find Only)")
        print(f"{'='*80}")
        for edge in sorted(component['transitive_edges'], key=lambda e: e['jaccard']):
            print(f"{edge['doc1_id']} <-> {edge['doc2_id']}: Jaccard={edge['jaccard']:.4f} (NO DIRECT MATCH)")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python view_component.py <component_file> <input_dir> <cc_id>")
        print("Example: python view_component.py component_analysis_results.jsonl /path/to/docs 3608")
        sys.exit(1)

    component_file = sys.argv[1]
    input_dir = sys.argv[2]
    cc_id = int(sys.argv[3])

    component = load_component(component_file, cc_id)
    if not component:
        print(f"Component {cc_id} not found in {component_file}")
        sys.exit(1)

    visualize_component(component, input_dir)
