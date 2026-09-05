"""
Batch evaluation harness: runs a fixed set of test queries against the search engine
to check ranking quality and query latency. See README.md for the queries that
performed poorly and the fixes that addressed them.
"""

import time
from search_engine import SearchEngine

TEST_QUERIES = [
    "irene gassko", "quantum", "information", "AI", "history",
    "master of software engineering", " machine learning", "cristina lopes",
    "ACM", "machine AND learning", "data AND structures", "M", "the", "uci research professor award",
    "cristina lopes information retrieval class 2026", "uci school of social sciences",
    "peter the anteater", "computer", "who teaches ICS6D", "ics student council",
    "honors", "research", "alfaro", "advanced",
]


def run_query(engine, query):
    start = time.time()
    results = engine.search(query)
    elapsed = time.time() - start
    print(f"Query: '{query}' | Results: {len(results)} | Time: {elapsed:.4f}s")
    for url in results[:5]:
        print(f"  - {url}")
    return results, elapsed


def run_evaluation():
    print("\n---------------- Search Engine ----------------\n")
    with SearchEngine() as engine:
        for query in TEST_QUERIES:
            print(query)
            run_query(engine, query)
    print("\n")


def demo():
    print("\n---------------- Search Engine ----------------\n")
    with SearchEngine() as engine:
        flag = True
        while flag:
            query = input("Query: ")
            print("\n")
            run_query(engine, query)
            print("\n")
            again = input("Search again? y/n ")
            if again.lower() == "n":
                flag = False


if __name__ == "__main__":
    run_evaluation()
    # demo()
