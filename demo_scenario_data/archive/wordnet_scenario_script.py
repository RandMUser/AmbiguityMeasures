import json
import argparse
from nltk.corpus import wordnet
import nltk
#python wordnet_scenario_script.py falcon bird aircraft jet fighter --scenarios scenario_data.json


# Ensure NLTK's WordNet is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_data(word_list, scenario_data):
    """
    Generate a lexical semantic dictionary from WordNet for a list of words.
    Incorporate additional senses from scenario data.
    
    Args:
        word_list (list): A list of words for which to retrieve lexical information.
        scenario_data (dict): Scenario data to augment the lexical information.
    
    Returns:
        dict: A dictionary with words as keys and their lexical information as values.
    """
    lexical_semantic_dict = {}
    
    for word in word_list:
        synsets = wordnet.synsets(word)
        word_data = []
        
        # Gather WordNet data
        for synset in synsets:
            definition = synset.definition()
            examples = synset.examples()
            lemmas = [lemma.name() for lemma in synset.lemmas()]
            hypernyms = [hypernym.name() for hypernym in synset.hypernyms()]
            hyponyms = [hyponym.name() for hyponym in synset.hyponyms()]
            
            word_data.append({
                'definition': definition,
                'examples': examples,
                'lemmas': lemmas,
                'hypernyms': hypernyms,
                'hyponyms': hyponyms
            })
        
        lexical_semantic_dict[word] = word_data
    
    # Incorporate scenario data to expand word senses
    for scenario in scenario_data['scenarios']:
        for query in scenario['queries']:
            word = query['lexical_topic']
            semantic_topic = query['semantic_topic']
            
            # Create additional word sense based on scenario information
            if word in lexical_semantic_dict:
                lexical_semantic_dict[word].append({
                    'definition': f"{word} related to {semantic_topic}",
                    'examples': [query['query']],
                    'lemmas': [word],
                    'hypernyms': [],
                    'hyponyms': []
                })
            else:
                lexical_semantic_dict[word] = [{
                    'definition': f"{word} related to {semantic_topic}",
                    'examples': [query['query']],
                    'lemmas': [word],
                    'hypernyms': [],
                    'hyponyms': []
                }]
    
    return lexical_semantic_dict

def save_to_json(data, output_file):
    """
    Save data to a JSON file.
    
    Args:
        data (dict): The data to save.
        output_file (str): The filename of the JSON output file.
    """
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Create a lexical semantic dictionary from WordNet for a given list of words.")
    parser.add_argument('words', metavar='W', type=str, nargs='+',
                        help='List of words to query WordNet')
    parser.add_argument('--output', type=str, default='lexical_semantic_dictionary.json',
                        help='Output JSON filename (default: lexical_semantic_dictionary.json)')
    parser.add_argument('--scenarios', type=str, required=True,
                        help='JSON file containing scenario data to incorporate')
    
    args = parser.parse_args()
    
    # Load scenario data
    with open(args.scenarios, 'r') as f:
        scenario_data = json.load(f)
    
    # Retrieve WordNet data for the given words
    wordnet_data = get_wordnet_data(args.words, scenario_data)
    
    # Save to JSON file
    save_to_json(wordnet_data, args.output)
    
    print(f"Lexical semantic dictionary saved to {args.output}")

if __name__ == '__main__':
    main()
