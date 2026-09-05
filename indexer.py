"""
Inverted index construction for a crawled HTML corpus.
"""

import os, re
import json
from collections import defaultdict
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
from nltk.stem import PorterStemmer
import warnings
import hashlib


class Posting:
    """A blueprint representing a token's occurrence inside a specific document."""

    def __init__(self, doc_id, tf=0, fields=None, positions=None, doc_length=0):
        self.doc_id = doc_id
        self.tf = tf
        self.fields = fields if fields is not None else []
        self.positions = positions if positions is not None else []
        self.doc_length = doc_length if doc_length is not None else 0

    def to_dict(self):
        """Transforms the object data layout into standard Python dictionaries so JSON can read it."""
        return {
            "doc_id": self.doc_id,
            "tf": self.tf,
            "fields": self.fields,
            "positions": self.positions,
            "doc_length": self.doc_length,
        }

    @classmethod
    def from_dict(cls, d):
        """A constructor method that recreates a native Posting object when reading JSON back from disk."""
        return cls(
            doc_id=d["doc_id"],
            tf=d["tf"],
            fields=d.get("fields", []),
            positions=d.get("positions", []),
            doc_length=d["doc_length"],
        )


class Indexer:
    """
    Builds a disk-backed inverted index from a directory of crawled JSON documents.

    Workflow:
    1.) Tokenize documents
    2.) Create list of unique tokens
    3.) For each token (key) append posting (vals) - builds inverted index

    A posting is the representation of the token's occurrence in a document. It contains
    the document id the token was found in, its (weighted) term frequency, the HTML fields
    it appeared in, and its token positions within the document.
    """

    FIELD_WEIGHTS = {
        "title": 10,
        "h1": 5,
        "h2": 4,
        "h3": 3,
        "b": 2,
        "strong": 2,
    }

    # Near-dupe fingerprints are split into 4 disjoint 16-bit bands (4 x 16 = 64 bits).
    # With a Hamming-distance threshold of 3, spreading at most 3 differing bits across
    # 4 bands guarantees (pigeonhole) at least one band matches exactly, so banding gives
    # the same recall as a full O(n) scan while only comparing against same-band candidates.
    NUM_BANDS = 4
    BAND_BITS = 16

    def __init__(
        self,
        root_directory="./DEV",
        final_postings_path="final_postings.txt",
        docmap_path="doc_id_map.json",
        seek_map_path="seek_map.json",
        partial_dir="partial_indexes",
        report_path="report.json",
        partial_flush_every=500,
    ):
        self.root_directory = root_directory
        self.final_postings_path = final_postings_path
        self.docmap_path = docmap_path
        self.seek_map_path = seek_map_path
        self.partial_dir = partial_dir
        self.report_path = report_path
        self.partial_flush_every = partial_flush_every

        self.stemmer = PorterStemmer()
        self.inverted_index = defaultdict(list)
        self.doc_id_map = {}

        # Exact (MD5) and near-duplicate (SimHash fingerprint) tracking.
        self.seen_content_hashes = set()
        self.fingerprints = {}  # fingerprint -> doc_id, for matched-doc lookup
        self.band_buckets = [dict() for _ in range(self.NUM_BANDS)]  # band value -> {fingerprints}

    # ---- Duplicate detection -------------------------------------------------

    @staticmethod
    def hash_helper_differences(h1, h2):
        """Computes Hamming distance (# of differing bits) between two hashes."""
        x = h1 ^ h2  # xor results in 1 for different bits
        dist = 0
        while x:
            dist += 1  # counting differing bits (1s)
            x &= x - 1  # removes least significant bit equal to 1 every iteration
        return dist

    @staticmethod
    def compute_fingerprint(tokens):
        """Computes a 64-bit fingerprint for a token list using the SimHash technique."""
        v = [0] * 64  # 64 bits
        for token in tokens:
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            for i in range(64):
                bit = (token_hash >> i) & 1
                if bit:
                    v[i] += 1
                else:
                    v[i] -= 1
        fingerprint = 0
        for i in range(64):
            if v[i] >= 0:  # majority vote
                fingerprint |= 1 << i
        return fingerprint

    def _band_value(self, fingerprint, band_index):
        shift = band_index * self.BAND_BITS
        mask = (1 << self.BAND_BITS) - 1
        return (fingerprint >> shift) & mask

    def is_near_dupe(self, new_fp, threshold=3):
        """
        Checks whether the new document's fingerprint is within `threshold` Hamming
        distance of any previously recorded fingerprint. Only candidates sharing a
        band with `new_fp` are compared (see NUM_BANDS/BAND_BITS above), instead of
        scanning every fingerprint seen so far.

        Checks candidates as they're found and returns on the first match, rather than
        first materializing the full cross-band candidate set - real near-duplicate
        clusters (e.g. near-identical wiki revision pages) can pile thousands of entries
        into one bucket, and the match is almost always the first candidate touched.
        """
        checked = set()
        for band_index in range(self.NUM_BANDS):
            band_val = self._band_value(new_fp, band_index)
            for fp in self.band_buckets[band_index].get(band_val, ()):
                if fp in checked:
                    continue
                checked.add(fp)
                if self.hash_helper_differences(new_fp, fp) <= threshold:
                    return True, self.fingerprints[fp]
        return False, None

    def record_fingerprint(self, fingerprint, doc_id):
        """Registers a fingerprint for future near-dupe lookups, in every band bucket."""
        self.fingerprints[fingerprint] = doc_id
        for band_index in range(self.NUM_BANDS):
            band_val = self._band_value(fingerprint, band_index)
            self.band_buckets[band_index].setdefault(band_val, set()).add(fingerprint)

    # ---- Parsing / tokenization -----------------------------------------------

    @staticmethod
    def read_input_files(file_path):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        return data["content"], data["url"]

    @staticmethod
    def parser(html_content):
        """
        Strips away non-readable programming markup logic like Javascript and CSS
        styling, exposing the readable string text layout used for near-duplicate
        calculation.
        """
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html_content, "lxml")
        for element in soup(["script", "style"]):
            element.decompose()
        return soup.get_text(separator=" ", strip=True)

    def process(self, text):
        """Standardizes an input text string into a list of stemmed tokens."""
        return [self.stemmer.stem(w) for w in re.findall(r"[a-zA-Z0-9]+", text.lower())]

    def extract_token_data(self, html_content):
        """
        Tokenizes documents using BeautifulSoup.
        Tokens: all alphanumeric sequences in the dataset.
        Stop words: none - all words are indexed, even the frequently occurring ones.
        Stemming: Porter stemming.
        Important words: words in bold, in headings (h1, h2, h3), and in titles are
        treated as more important than the other words via FIELD_WEIGHTS.
        """
        soup = BeautifulSoup(html_content, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()  # Discards script sections

        data = {}
        position = 0

        def add_tokens(text, field, boost):
            """
            Internal helper tracking global token offsets. If a word repeats, it
            increments its boosted tf, stores the tag label, and logs the sequential
            integer position.
            """
            nonlocal position
            for tok in self.process(text):
                if tok not in data:
                    data[tok] = {"tf": 0, "fields": [], "positions": []}
                data[tok]["tf"] += boost
                if field not in data[tok]["fields"]:
                    data[tok]["fields"].append(field)
                data[tok]["positions"].append(position)
                position += 1

        # <title, h1-h3, b, strong> - adds boosts as needed
        for tag_name, weight in self.FIELD_WEIGHTS.items():
            for element in soup.find_all(tag_name):
                text = element.get_text(separator=" ", strip=True)
                if text:
                    add_tokens(text, field=tag_name, boost=weight)
                element.decompose()  # avoids double-counting when body text is gathered below

        # Gathers remaining text with default weight of 1
        remaining_body_text = soup.get_text(separator=" ", strip=True)
        if remaining_body_text:
            add_tokens(remaining_body_text, field="body", boost=1)

        return data

    def build_index(self, processed_tokens, doc_id):
        """Converts extracted tokens into Posting instances and adds them to the in-memory index."""
        doc_length = sum(info["tf"] for info in processed_tokens.values())
        for token, info in processed_tokens.items():
            posting = Posting(
                doc_id=doc_id,
                tf=info["tf"],
                fields=info["fields"],
                positions=info["positions"],
                doc_length=doc_length,
            )
            self.inverted_index[token].append(posting)

    # ---- Partial index flush / merge -------------------------------------------

    def flush_partial(self, part_num):
        """Converts Posting objects to dictionaries and sorts tokens alphabetically before flushing to disk."""
        os.makedirs(self.partial_dir, exist_ok=True)
        path = os.path.join(self.partial_dir, f"part_{part_num:04d}.json")
        serialisable = {
            tok: [p.to_dict() for p in postings]
            for tok, postings in sorted(self.inverted_index.items())
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f)
        print(f"  Flushed partial #{part_num} ({len(serialisable)} tokens) -> {path}")
        return path

    @staticmethod
    def merge_partials(partial_paths):
        """Reads all partial index files back from disk, recombining their postings into one index."""
        merged = defaultdict(list)
        for path in partial_paths:
            with open(path, "r", encoding="utf-8") as f:
                part = json.load(f)
            for token, raw_postings in part.items():
                merged[token].extend(Posting.from_dict(p) for p in raw_postings)
        return merged

    def write_report(self, num_docs, num_tokens):
        postings_size = os.path.getsize(self.final_postings_path)
        seek_map_size = os.path.getsize(self.seek_map_path)
        total_index_kb = (postings_size + seek_map_size) / 1024

        report = {
            "Number of indexed documents": num_docs,
            "Number of unique tokens": num_tokens,
            "Total index size (KB)": round(total_index_kb, 2),
        }

        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print("\n Report:")
        for k, v in report.items():
            print(f" {k}: {v}")
        return report

    # ---- Top-level pipeline -----------------------------------------------------

    def collect_document_paths(self):
        """Walks the source content directory to collect all valid target document paths."""
        all_files = []
        for root, _, files in os.walk(self.root_directory):
            for fname in files:
                if fname.endswith(".json"):
                    all_files.append(os.path.join(root, fname))
        return all_files

    def run(self):
        all_files = self.collect_document_paths()
        print(f"Found {len(all_files)} documents.")

        partial_paths = []
        part_num = 0
        num_docs = 0

        for file_path in all_files:
            try:
                html_content, url = self.read_input_files(file_path)
            except Exception as e:
                print(f"  [SKIP - read error] {file_path}: {e}")
                continue

            # Exact duplicate check
            page_hash = hashlib.md5(html_content.encode("utf-8")).hexdigest()
            if page_hash in self.seen_content_hashes:
                print(f"  [SKIP - Exact Duplicate] {url}")
                continue
            self.seen_content_hashes.add(page_hash)

            # Near duplicate check
            try:
                clean_text = self.parser(html_content)
                query_tokens = self.process(clean_text)
            except Exception as e:
                print(f"  [SKIP - Pre-parsing error] {file_path}: {e}")
                continue

            if not query_tokens:
                continue

            new_fp = self.compute_fingerprint(query_tokens)
            near_duplicate_found, matching_doc = self.is_near_dupe(new_fp, threshold=3)
            if near_duplicate_found:
                print(f"  [SKIP - Near Duplicate of Doc #{matching_doc}] {url}")
                continue
            self.record_fingerprint(new_fp, num_docs)

            # Assigns integer ID to the verified URL, processes tag token
            # extractions, and builds the index collection layout.
            doc_id = num_docs
            self.doc_id_map[doc_id] = url

            try:
                token_data = self.extract_token_data(html_content)
            except Exception as e:
                print(f"  [SKIP - parse error] {file_path}: {e}")
                num_docs += 1
                continue

            self.build_index(token_data, doc_id)
            num_docs += 1

            if num_docs % self.partial_flush_every == 0:
                path = self.flush_partial(part_num)
                partial_paths.append(path)
                self.inverted_index.clear()
                part_num += 1
                print(f"  Progress: {num_docs} / {len(all_files)}")

        if self.inverted_index:
            path = self.flush_partial(part_num)
            partial_paths.append(path)
            self.inverted_index.clear()

        print(f"\nMerging {len(partial_paths)} partial file(s)...")
        final_index = self.merge_partials(partial_paths)

        # Partial files are only scratch space for the merge step; drop them once
        # merged so disk usage doesn't stay at ~2x the final index size.
        for path in partial_paths:
            os.remove(path)
        if os.path.isdir(self.partial_dir) and not os.listdir(self.partial_dir):
            os.rmdir(self.partial_dir)

        print(f"Writing to {self.final_postings_path}...")
        seek_map = {}
        with open(self.final_postings_path, "w", encoding="utf-8") as f_out:
            for token, postings in sorted(final_index.items()):
                postings_str = json.dumps([p.to_dict() for p in postings])
                offset = f_out.tell()  # byte position where this line starts, for O(1) disk lookup
                f_out.write(postings_str + "\n")
                seek_map[token] = [offset, len(postings_str)]

        print(f"Writing lookup map to {self.seek_map_path}...")
        with open(self.seek_map_path, "w", encoding="utf-8") as f_seek:
            json.dump(seek_map, f_seek)

        print(f"Doc ID map -> {self.docmap_path}")
        with open(self.docmap_path, "w", encoding="utf-8") as f_doc:
            json.dump(self.doc_id_map, f_doc, indent=2)

        report = self.write_report(num_docs, len(seek_map))
        return report


def main():
    import nltk

    nltk.download("punkt_tab")
    indexer = Indexer()
    indexer.run()


if __name__ == "__main__":
    main()
