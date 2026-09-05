"""
Query engine for the disk-backed inverted index built by indexer.py.
"""

import json
import math
import re

from nltk.stem import PorterStemmer


class SearchEngine:
    """
    Query engine over a disk-backed inverted index built by indexer.Indexer.

    Postings are fetched lazily via a byte-offset seek map, so the full index
    never needs to be loaded into memory - only the (much smaller) seek map
    and doc id map are.
    """

    def __init__(
        self,
        postings_path="final_postings.txt",
        seek_map_path="seek_map.json",
        docmap_path="doc_id_map.json",
    ):
        self.postings_path = postings_path
        self.seek_map_path = seek_map_path
        self.docmap_path = docmap_path

        self.stemmer = PorterStemmer()
        self.seek_map = {}
        self.doc_map = {}
        self._postings_file = None

    # ---- Setup / teardown -------------------------------------------------

    def load(self):
        """Loads the seek map + doc id map and opens the postings file for random access."""
        print("Initializing Search Engine")
        with open(self.seek_map_path, "r", encoding="utf-8") as f:
            self.seek_map = json.load(f)
        with open(self.docmap_path, "r", encoding="utf-8") as f:
            self.doc_map = json.load(f)
        self._postings_file = open(self.postings_path, "r", encoding="utf-8")
        return self

    def close(self):
        if self._postings_file is not None:
            self._postings_file.close()
            self._postings_file = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---- Postings access ----------------------------------------------------

    def fetch_postings_from_disk(self, token):
        """Jumps to a token's postings line on disk in O(1) instead of scanning the file."""
        if token not in self.seek_map:
            return []
        offset, length = self.seek_map[token]
        self._postings_file.seek(offset)
        return json.loads(self._postings_file.read(length))

    @staticmethod
    def get_postings(token, index):
        """Uses a local {token: postings} index to retrieve the set of matching doc ids."""
        unique_doc_ids = set()
        if token in index:
            for posting in index[token]:
                unique_doc_ids.add(posting["doc_id"])
        return unique_doc_ids

    def bool_representation(self, tokens, index):
        """Turns a token list into an AND-intersected set of candidate doc ids."""
        if not tokens:
            return set()
        result_doc_ids = self.get_postings(tokens[0], index)
        for token in tokens[1:]:
            if not result_doc_ids:
                break
            result_doc_ids = result_doc_ids.intersection(self.get_postings(token, index))
        if not result_doc_ids:
            return []
        return result_doc_ids

    # ---- Query processing -----------------------------------------------------

    def process_query(self, query):
        """Tokenizes + stems a raw query string, de-duplicating while preserving order."""
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        stemmed = [self.stemmer.stem(w) for w in words]
        tokens = []
        for word in stemmed:
            if word not in tokens:
                tokens.append(word)
        return tokens

    # ---- Baseline TF-IDF scorer -------------------------------------------------

    def assign_scores(self, tokens, result_doc_ids, index, total_docs):
        """
        Baseline TF-IDF scorer (docs-frequency-capped IDF x log-tf weight). `search()`
        below uses a refined version of this that additionally applies field-boosting
        and phrase-proximity scoring; this method is kept as the simpler, directly
        testable scoring building block.
        """
        doc_scores = {doc_id: 0.0 for doc_id in result_doc_ids}
        for token in tokens:
            if token not in index:
                continue

            docs_with_token = len(index[token])

            # Skip tokens appearing in more than 30% of all docs (stop-word-like terms).
            if docs_with_token / total_docs > 0.30:
                continue

            idf = math.log10(total_docs / docs_with_token)

            for posting in index[token]:
                d_id = posting["doc_id"]
                if d_id in doc_scores:
                    raw_tf = posting["tf"]
                    tf_weight = (1 + math.log10(raw_tf)) / math.log10(1 + posting["doc_length"])
                    doc_scores[d_id] += tf_weight * idf
        return doc_scores

    @staticmethod
    def score_sort(doc_scores):
        """Returns doc ids sorted by score, highest first."""
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in sorted_docs]

    @staticmethod
    def fast_proximity_check(pos1, pos2, window=20):
        """Two-pointer scan checking whether any positions in pos1/pos2 fall within `window`."""
        i, j = 0, 0
        while i < len(pos1) and j < len(pos2):
            if abs(pos1[i] - pos2[j]) <= window:
                return True
            if pos1[i] < pos2[j]:
                i += 1
            else:
                j += 1
        return False

    # ---- Main search pipeline -----------------------------------------------------

    def search(self, query, top_k=5):
        tokens = self.process_query(query)
        if not tokens:
            return []

        # Builds a local index map for just the query terms.
        index = {}
        docs_by_token = {}
        for t in tokens:
            postings_data = self.fetch_postings_from_disk(t)
            index[t] = postings_data
            docs_by_token[t] = {p["doc_id"] for p in postings_data}

        # AND
        candidate_docs = docs_by_token[tokens[0]]
        for t in tokens[1:]:
            candidate_docs = candidate_docs & docs_by_token[t]

        # Fallback to OR if AND yields nothing (handles rare/absent terms in the query).
        fallback = False
        if not candidate_docs:
            fallback = True
            candidate_docs = set()
            for t in tokens:
                candidate_docs = candidate_docs | docs_by_token[t]

        if not candidate_docs:
            return []

        # TF-IDF scoring with field boosts.
        total_docs = len(self.doc_map)
        scores = {doc_id: 0.0 for doc_id in candidate_docs}

        for t in tokens:
            postings = index.get(t, [])
            df = len(postings)
            if df == 0:
                continue

            idf = math.log10(total_docs / df)

            for p in postings:
                d_id = p["doc_id"]
                if d_id in candidate_docs:
                    weight_boost = 1.0
                    fields = p.get("fields", [])
                    if "title" in fields:
                        weight_boost += 3.0
                    if any(h in fields for h in ["h1", "h2", "h3"]):
                        weight_boost += 1.5
                    tf_raw = p["tf"]
                    tf_norm = 1 + math.log10(tf_raw)
                    scores[d_id] += tf_norm * idf * weight_boost

        # Phrase-proximity boost: reward docs where query terms co-occur nearby.
        if len(tokens) > 1 and not fallback:
            lookup_positions = {t: {} for t in tokens}
            for t in tokens:
                for p in index.get(t, []):
                    d_id = p["doc_id"]
                    if d_id in candidate_docs:
                        lookup_positions[t][d_id] = p.get("positions", [])

            for d in candidate_docs:
                pair_matches = 0
                for i in range(len(tokens)):
                    for j in range(i + 1, len(tokens)):
                        pos1 = lookup_positions[tokens[i]].get(d, [])
                        pos2 = lookup_positions[tokens[j]].get(d, [])
                        if pos1 and pos2 and self.fast_proximity_check(pos1, pos2, window=20):
                            pair_matches += 1
                if pair_matches > 0:
                    scores[d] += 2.5 * pair_matches  # scaled to the tf-idf score range (0.5 - 3.0)

        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        urls = []
        for doc_id, _ in sorted_docs[:top_k]:
            str_id = str(doc_id)
            if str_id in self.doc_map:
                urls.append(self.doc_map[str_id])
        return urls


def main():
    test_queries = [
        "cristina lopes",
        "machine learning",
        "ACM",
        "master of software engineering",
    ]

    print("loading")
    with SearchEngine() as engine:
        for query in test_queries:
            print(f"Query: {query}")
            results = engine.search(query)
            print(f"Num results: {len(results)}")
            print(f"Results: {results[:5]}")
            for url in results[:5]:
                print(f" - {url}")


if __name__ == "__main__":
    main()
