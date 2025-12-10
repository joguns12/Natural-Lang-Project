# evaluate_accuracy.py
# Test pronoun resolution accuracy on 35 random sentences, output results to CSV

import spacy
import coreferee
import pandas as pd
import re

def load_model():
    # Load spaCy model with coreference resolution
    print("Loading SpaCy + Coreferee...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    return nlp

def resolve_pronouns_with_details(nlp, sentence, pronoun, candidates):
    # Resolve pronouns and return resolved text and candidate index
    doc = nlp(sentence)
    resolved_text = sentence
    resolved_candidate_idx = None
    
    if doc._.coref_chains:
        for chain in doc._.coref_chains:
            # find the main referent for this coreference chain
            main_idx = chain.most_specific_mention_index
            main_mention = chain.mentions[main_idx]
            main_text = doc[main_mention[0]: main_mention[-1] + 1].text

            for mention in chain.mentions:
                if mention == main_mention:
                    continue
                mention_text = doc[mention[0]: mention[-1] + 1].text
                
                # check if this mention is the target pronoun
                if mention_text.lower() == pronoun.lower():
                    resolved_text = resolved_text.replace(mention_text, main_text)
                    # determine which candidate the pronoun resolved to
                    for i, candidate in enumerate(candidates):
                        if candidate.lower() in main_text.lower() or main_text.lower() in candidate.lower():
                            resolved_candidate_idx = i
                            break
                else:
                    # replace other mentions in chain
                    resolved_text = resolved_text.replace(mention_text, main_text)
    
    return resolved_text, resolved_candidate_idx

def evaluate_accuracy():
    # Evaluate pronoun resolution accuracy on the entire dpr_train.csv
    nlp = load_model()
    
    print(f"\nLoading dataset...")
    df = pd.read_csv('dpr_train.csv')
    sample = df  # use all rows
    
    correct = 0
    total = 0
    results = []
    
    print(f"\nEvaluating on {len(sample)} sentences...")
    for idx, (_, row) in enumerate(sample.iterrows(), 1):
        sentence = row["sentence"]
        pronoun = row["pronoun"]
        candidates = eval(row["candidates"])
        label = row["label"]
        resolved_text, resolved_idx = resolve_pronouns_with_details(nlp, sentence, pronoun, candidates)
        is_correct = resolved_idx == label
        if is_correct:
            correct += 1
        total += 1
        results.append({
            'sentence': sentence,
            'pronoun': pronoun,
            'candidates': candidates,
            'ground_truth_idx': label,
            'resolved_idx': resolved_idx,
            'correct': is_correct,
            'resolved_text': resolved_text
        })
    # ...existing code...
    
    print("\n" + "=" * 100)
    # calculate accuracy percentage
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\n[ACCURACY RESULTS]")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"\nEvaluation complete. Results saved to 'accuracy_results.csv'")
    
    # export detailed results to CSV for further analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv('accuracy_results.csv', index=False)
    
    return accuracy, results_df

if __name__ == "__main__":
    evaluate_accuracy()
