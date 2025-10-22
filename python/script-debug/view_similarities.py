#!/usr/bin/env python3
"""
Interactive viewer for pairwise document similarity scores.
Displays documents side-by-side with their similarity scores.
"""

import json
import sys
import tty
import termios
from typing import List, Dict


def load_similarities(filepath: str) -> List[Dict]:
    """Load and sort similarities by descending Jaccard score."""
    similarities = []
    with open(filepath, 'r') as f:
        for line in f:
            similarities.append(json.loads(line))

    # Sort by descending jaccard_score
    similarities.sort(key=lambda x: x['jaccard_score'], reverse=True)
    return similarities


def wrap_text(text: str, width: int) -> List[str]:
    """Wrap text to specified width, breaking on spaces when possible."""
    lines = []
    words = text.split()
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + len(current_line) <= width:
            current_line.append(word)
            current_length += word_len
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_len

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def display_pair(similarity: Dict, pair_num: int, total_pairs: int, width: int = 200):
    """Display a single pair of documents side-by-side."""
    # Clear screen
    print('\033[2J\033[H', end='')

    # Calculate column width (leave some margin)
    col_width = (width - 5) // 2

    # Header
    print("=" * width)
    score = similarity['jaccard_score']
    header = f"Similarity: {score:.4f}  (Pair {pair_num + 1} of {total_pairs})"
    print(header.center(width))
    print("=" * width)
    print()

    # Document IDs
    doc1_id = similarity.get('doc1_id', 'Unknown')
    doc2_id = similarity.get('doc2_id', 'Unknown')

    print(f"Doc 1: {doc1_id[:col_width-7]:<{col_width-7}} | Doc 2: {doc2_id[:col_width-7]}")
    print()

    # Top border
    print(f"┌{'─' * col_width}┬{'─' * col_width}┐")

    # START sections
    doc1_start = similarity.get('doc1_text_start', '')
    doc2_start = similarity.get('doc2_text_start', '')

    lines1_start = wrap_text(doc1_start, col_width - 2)
    lines2_start = wrap_text(doc2_start, col_width - 2)

    print(f"│ {'DOC 1 - START':<{col_width-2}} │ {'DOC 2 - START':<{col_width-2}} │")
    print(f"├{'─' * col_width}┼{'─' * col_width}┤")

    max_start_lines = max(len(lines1_start), len(lines2_start))
    for i in range(max_start_lines):
        left = lines1_start[i] if i < len(lines1_start) else ''
        right = lines2_start[i] if i < len(lines2_start) else ''
        print(f"│ {left:<{col_width-2}} │ {right:<{col_width-2}} │")

    # Middle separator
    print(f"├{'─' * col_width}┼{'─' * col_width}┤")

    # END sections
    doc1_end = similarity.get('doc1_text_end', '')
    doc2_end = similarity.get('doc2_text_end', '')

    lines1_end = wrap_text(doc1_end, col_width - 2)
    lines2_end = wrap_text(doc2_end, col_width - 2)

    print(f"│ {'DOC 1 - END':<{col_width-2}} │ {'DOC 2 - END':<{col_width-2}} │")
    print(f"├{'─' * col_width}┼{'─' * col_width}┤")

    max_end_lines = max(len(lines1_end), len(lines2_end))
    for i in range(max_end_lines):
        left = lines1_end[i] if i < len(lines1_end) else ''
        right = lines2_end[i] if i < len(lines2_end) else ''
        print(f"│ {left:<{col_width-2}} │ {right:<{col_width-2}} │")

    # Bottom border
    print(f"└{'─' * col_width}┴{'─' * col_width}┘")
    print()

    # Navigation help
    print("[n]ext  [p]revious  [j]ump to index  [q]uit")


def get_key():
    """Get a single keypress from the user."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 view_similarities.py <similarities.jsonl>")
        sys.exit(1)

    filepath = sys.argv[1]

    print("Loading similarities...")
    similarities = load_similarities(filepath)
    print(f"Loaded {len(similarities)} similarity pairs")

    if not similarities:
        print("No similarities found!")
        sys.exit(1)

    current_idx = 0

    while True:
        display_pair(similarities[current_idx], current_idx, len(similarities))

        key = get_key()

        if key == 'q':
            break
        elif key == 'n':
            current_idx = min(current_idx + 1, len(similarities) - 1)
        elif key == 'p':
            current_idx = max(current_idx - 1, 0)
        elif key == 'j':
            # Jump to specific index
            print("\nEnter index (0-{}): ".format(len(similarities) - 1), end='', flush=True)
            # Restore terminal settings temporarily to get input
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

            try:
                idx_str = input()
                idx = int(idx_str)
                if 0 <= idx < len(similarities):
                    current_idx = idx
            except (ValueError, EOFError):
                pass

    # Clear screen on exit
    print('\033[2J\033[H', end='')
    print("Goodbye!")


if __name__ == '__main__':
    main()
