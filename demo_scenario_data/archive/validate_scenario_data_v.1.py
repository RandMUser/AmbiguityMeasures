import json

def validate_scenarios(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    for scenario in scenarios:
        print(f"Validating Scenario {scenario['scenario_id']}: {scenario['title']}")
        queries = {q['qid']: q for q in scenario.get("queries", [])}
        passages = {p['pid']: p for p in scenario.get("passages", [])}
        qrels = scenario.get("qrels", [])

        for qrel in qrels:
            qid = qrel['qid']
            pid = qrel['pid']
            relevance = qrel['relevance']

            query = queries.get(qid)
            passage = passages.get(pid)

            if not query or not passage:
                print(f"  - Error: Query ID {qid} or Passage ID {pid} not found in scenario.")
                continue

            query_lexical = query['lexical_topic']
            query_semantic = query['semantic_topic']
            passage_lexical = passage['lexical_topic']
            passage_semantic = passage['semantic_topic']

            if relevance == 'relevant':
                if query_lexical != passage_lexical or query_semantic != passage_semantic:
                    print(f"  - Warning: Relevant QID {qid} and PID {pid} have mismatched topics.")
                    print(f"    Query (Lexical: {query_lexical}, Semantic: {query_semantic})")
                    print(f"    Passage (Lexical: {passage_lexical}, Semantic: {passage_semantic})")
            elif relevance == 'not-relevant':
                if query_lexical == passage_lexical and query_semantic == passage_semantic:
                    print(f"  - Warning: Not-relevant QID {qid} and PID {pid} have matching topics.")
                    print(f"    Query (Lexical: {query_lexical}, Semantic: {query_semantic})")
                    print(f"    Passage (Lexical: {passage_lexical}, Semantic: {passage_semantic})")

if __name__ == "__main__":
    # Replace 'scenario_data.json' with the actual path to your scenario JSON file
    validate_scenarios('scenario_data.json')
