import json
from collections import defaultdict

# Load the JSON data from a file
def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# Print queries, documents, and qrel relationships in a human-readable format
def print_scenarios(data):
    for scenario in data['scenarios']:
        print(f"\nScenario ID: {scenario['scenario_id']}")
        print(f"Title: {scenario['title']}")
        print(f"Intuition: {scenario['intuition']}\n")
        
        print("Queries:")
        for query in scenario['queries']:
            print(f"  Query ID: {query['qid']}")
            print(f"  Query: {query['query']}")
            print(f"  Lexical Topic: {query['lexical_topic']}")
            print(f"  Semantic Topic: {query['semantic_topic']}\n")
        
        print("Passages:")
        for passage in scenario['passages']:
            print(f"  Passage ID: {passage['pid']}")
            print(f"  Text: {passage['text']}")
            print(f"  Lexical Topic: {passage['lexical_topic']}")
            print(f"  Semantic Topic: {passage['semantic_topic']}\n")
        
        print("QRELs:")
        for qrel in scenario['qrels']:
            print(f"  Query ID: {qrel['qid']}, Passage ID: {qrel['pid']}, Relevance: {qrel['relevance']}")

# Print lexical_semantic_dictionary in a human-readable format
def print_lexical_semantic_dictionary(data):
    print("\nLexical Semantic Dictionary:")
    for entry in data['lexical_semantic_dictionary']:
        print(f"\nTerm: {entry['term']}")
        for sense in entry['senses']:
            print(f"  Definition: {sense['definition']}")
            print(f"  Examples: {', '.join(sense['examples']) if sense['examples'] else 'None'}")
            print(f"  Lemmas: {', '.join(sense['lemmas'])}")
            print(f"  Hypernyms: {', '.join(sense['hypernyms']) if sense['hypernyms'] else 'None'}")
            print(f"  Hyponyms: {', '.join(sense['hyponyms']) if sense['hyponyms'] else 'None'}")
            print(f"  Semantic Topics: {', '.join(sense['semantic_topics']) if sense['semantic_topics'] else 'None'}\n")

# Organize the data into a data structure that can be used for further processing
def organize_data(data):
    organized_data = defaultdict(lambda: defaultdict(list))
    
    # Organize scenarios
    for scenario in data['scenarios']:
        organized_data['scenarios'][scenario['scenario_id']] = {
            'title': scenario['title'],
            'intuition': scenario['intuition'],
            'queries': scenario['queries'],
            'passages': scenario['passages'],
            'qrels': scenario['qrels']
        }
    
    # Organize lexical semantic dictionary
    for entry in data['lexical_semantic_dictionary']:
        organized_data['lexical_semantic_dictionary'][entry['term']] = entry['senses']
    
    return organized_data

# Main function to execute the script
def main():
    #demo_scenario_data/lexical_semantic_scenario.json
    filename = 'lexical_semantic_scenario.json'  # Replace with your JSON filename
    data = load_json(filename)
    
    print_scenarios(data)
    print_lexical_semantic_dictionary(data)
    organized_data = organize_data(data)
    
    

    # Just an example of how to use the organized data
    print("\nOrganized Data Summary:")
    num_scenarios = len(organized_data['scenarios'])
    print(f"Number of Scenarios: {num_scenarios}")
    total_queries = sum([len(scenario_data['queries']) for scenario_data in organized_data['scenarios'].values() ])
    avg_queries = total_queries / num_scenarios
    print(f"Average queries per Scenario: {avg_queries}")
    total_docs = sum([len(scenario_data['passages']) for scenario_data in organized_data['scenarios'].values() ])
    avg_docs = total_docs / num_scenarios
    print(f"Average documents(or passages) per Scenario: {avg_docs}")
    print(f"Number of Lexical Semantic Dictionary Entries: {len(organized_data['lexical_semantic_dictionary'])}")

if __name__ == "__main__":
    main()
