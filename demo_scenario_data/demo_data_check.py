import json

word_sense_issue = {} #term is the key, value is the missing word sense...

def validate_scenarios(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    lexical_semantic_dict = {entry['term']: entry for entry in data.get("lexical_semantic_dictionary", [])}
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

            # Validate that the lexical and semantic terms are defined in the dictionary
            if query_lexical not in lexical_semantic_dict:
                word_sense_issue[query_lexical] = {"Issue Source":"query_lexical", "Issue Term":query_lexical, "Issue":"Not in lexical_semantic_dict"}
                #This is a big problem and prevents diving into the identification of word senses for this 'lemma'.
            
            if query_semantic not in lexical_semantic_dict:
                word_sense_issue[query_lexical] = {"Issue Source":"query_lexical", "Issue Term":query_lexical, "Issue":"Not in lexical_semantic_dict"}

            if query_lexical not in lexical_semantic_dict or passage_lexical not in lexical_semantic_dict:
                print(f"  - Error: Lexical term '{query_lexical}' or '{passage_lexical}' not found in lexical-semantic dictionary.")
                continue

            query_possible_senses = lexical_semantic_dict[query_lexical].get("senses", [])
            #query_lexical is something like 'falcon'
            # TODO: Stopping poing 22 Oct 2024 at 1928, going home for dinner.
            query_possible_semantic_topics = []
            for word_sense in query_possible_senses:
                for st in word_sense['semantic_topics']:
                    query_possible_senses.append(st)


            passage_senses = lexical_semantic_dict[passage_lexical].get("senses", [])

            query_sense_match = any(sense['definition'] == query_semantic for sense in query_possible_senses)
            passage_sense_match = any(sense['definition'] == passage_semantic for sense in passage_senses)

            if not query_sense_match or not passage_sense_match:
                print(f"  - Error: Semantic topic '{query_semantic}' or '{passage_semantic}' not found in lexical-semantic dictionary for '{query_lexical}' or '{passage_lexical}'.")
                continue

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
    validate_scenarios('lexical_semantic_scenario.json')
