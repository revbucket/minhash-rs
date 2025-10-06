""" Some code that programmatically generates some test-data for deduplication examples """
import argparse
import json
import os

from lorem_text import lorem  # !pip install lorem-text


def make_exact_example_docs(target_dir, multi=False):
    """
    Strategy here is easy:
            Make 25 different "texts", and put them in 25 different files
            where the i^th file has documents (0..i)

            Output (after exact dedup) should be easily verified to just have a unique 'text' field across all output files
    """
    os.makedirs(target_dir, exist_ok=True)
    if multi:
        for i in range(4):
            subdir = os.path.join(target_dir, "subdir_%02d" % i)
            os.makedirs(subdir, exist_ok=True)

    doc_id = 0  # unique doc ID
    for i in range(25):
        if not multi:
            output_dir = target_dir
        else:
            output_dir = os.path.join(target_dir, "subdir_%02d" % (i % 4))

        output_dir = (
            target_dir
            if multi == False
            else os.path.join(target_dir, "subdir_%02d" % (i % 4))
        )
        file_contents = []
        for j in range(i + 1):
            file_contents.append(
                {"id": "id_%04d" % doc_id, "text": "this doc has content: %02d" % j}
            )
            doc_id += 1
        with open(os.path.join(output_dir, "example_file_%02d.jsonl" % i), "w") as f:
            f.write("\n".join(json.dumps(doc) for doc in file_contents))


def make_fuzzy_example_docs(target_dir, multi=False):
    """
    Strategy here is more complex.
            Make 25 different lorem ipsum texts of sufficient length, and make 25 different files.
            Append these lorem texts with an easily identifiable tag, like "lorem ipsum ... LOREM_01"
            For the i^th file, put "file_{FILENUM} {lorem_j}" for j in (0..i)

    Output should be verifiable by noting that only one LOREM_XX at the end of each text should be present
    """
    doc_id = 0  # doc ID

    # Make 25 different lorem ipsum style docs
    lorem_docs = {
        i: " ".join(lorem.paragraph() for _ in range(10)) + " LOREM_%02d" % i
        for i in range(25)
    }

    os.makedirs(target_dir, exist_ok=True)
    if multi:
        for i in range(4):
            subdir = os.path.join(target_dir, "subdir_%02d" % i)
            os.makedirs(subdir, exist_ok=True)
    doc_id = 0
    for i in range(25):
        output_dir = (
            target_dir
            if not multi
            else os.path.join(target_dir, "subdir_%02d" % (i % 4))
        )
        file_contents = []
        for j in range(i + 1):
            lorem_doc = lorem_docs[j]
            file_contents.append(
                {"id": "id_%04d" % doc_id, "text": "FILE_%02d %s" % (i, lorem_doc)}
            )
            doc_id += 1
        with open(os.path.join(output_dir, "example_file_%02d.jsonl" % i), "w") as f:
            f.write("\n".join(json.dumps(doc) for doc in file_contents))


def main(target_dir, fuzzy=False, multi=False):
    if fuzzy:
        make_fuzzy_example_docs(target_dir, multi=multi)
    else:
        make_exact_example_docs(target_dir, multi=multi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate test data for deduplication examples"
    )
    parser.add_argument(
        "--target-dir",
        required=True,
        help="Directory where test files will be generated",
    )
    parser.add_argument(
        "--fuzzy",
        action="store_true",
        help="Generate fuzzy deduplication test data (default: exact deduplication)",
    )

    parser.add_argument(
        "--multi",
        action="store_true",
        help="If present, split the data into 4 subdirectories to simulate a multinode setting",
    )

    args = parser.parse_args()
    main(args.target_dir, args.fuzzy, args.multi)
