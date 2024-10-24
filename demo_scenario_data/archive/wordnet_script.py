import json
import argparse
from nltk.corpus import wordnet
import nltk


#python wordnet_script.py falcon appple fruit bird technology company --output lexical_data.json

#python wordnet_script.py falcon bird aircraft jet fighter --output lexical_data.json

#python wordnet_script.py word1 word2 word3 --output lexical_data.json
#pip install nltk
# nltk_wordnet_lexical_dictionary


# Ensure NLTK's WordNet is downloaded
nltk.download('wordnet')
nltk.download('omw-1.4')

def get_wordnet_data(word_list):
    """
    Generate a lexical semantic dictionary from WordNet for a list of words.
    
    Args:
        word_list (list): A list of words for which to retrieve lexical information.
    
    Returns:
        dict: A dictionary with words as keys and their lexical information as values.
    """
    lexical_semantic_dict = {}
    
    for word in word_list:
        synsets = wordnet.synsets(word)
        word_data = []
        
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
    
    args = parser.parse_args()
    
    # Retrieve WordNet data for the given words
    wordnet_data = get_wordnet_data(args.words)
    
    # Save to JSON file
    save_to_json(wordnet_data, args.output)
    
    print(f"Lexical semantic dictionary saved to {args.output}")

if __name__ == '__main__':
    main()
