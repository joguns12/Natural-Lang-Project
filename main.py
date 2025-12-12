# main.py
# CLI hub for pronoun resolution, accuracy testing, and LDA topic modeling comparison

import sys
import os

def show_menu():
    # Display the main menu
    print("\n" + "="*100)
    print("PRONOUN RESOLUTION & TOPIC MODELING TOOLKIT")
    print("="*100)
    print("\nChoose an operation:")
    print("1. Interactive Pronoun Resolution (manual or dataset)")
    print("2. Evaluate Accuracy (test on 35 sentences)")
    print("3. Compare LDA Topic Models (raw vs. pronoun-resolved)")
    print("4. Exit")
    print("\n" + "-"*100)
    choice = input("Enter choice (1-4): ").strip()
    return choice

def run_interactive():
    # Run interactive pronoun resolution
    print("\n" + "="*100)
    print("INTERACTIVE PRONOUN RESOLUTION")
    print("="*100)
    
    import spacy
    import coreferee
    import pandas as pd
    
    def load_model():
        print("Loading SpaCy + Coreferee...")
        nlp = spacy.load("en_core_web_sm")
        # add coreferee pipeline for coreference resolution
        nlp.add_pipe("coreferee")
        return nlp
    
    def resolve_pronouns(nlp, paragraph):
        doc = nlp(paragraph)
        if doc._.coref_chains:
            resolved_text = paragraph
            # iterate through each coreference chain
            for chain in doc._.coref_chains:
                # get the most specific mention as the referent
                main_idx = chain.most_specific_mention_index
                main_mention = chain.mentions[main_idx]
                main_text = doc[main_mention[0]: main_mention[-1] + 1].text
                
                # replace all other mentions in the chain with the main referent
                for mention in chain.mentions:
                    if mention == main_mention:
                        continue
                    mention_text = doc[mention[0]: mention[-1] + 1].text
                    resolved_text = resolved_text.replace(mention_text, main_text)
            return resolved_text
        else:
            return paragraph
    
    def run_dataset_mode(nlp, file_path="dpr_train.csv", sample_size=5):
        resolved_results = []

        print(f"\nLoading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        # sample random examples from dataset
        if sample_size is None or sample_size > len(df):
            # if sample size too large, shuffle entire dataset
            examples = df.sample(frac=1)
        else:
            examples = df.sample(sample_size)
        
        # display each example with its resolution
        for _, row in examples.iterrows():
            sentence = row["sentence"]
            pronoun = row["pronoun"]
            candidates = row["candidates"]
            print("\n" + "-"*100)
            print(f"Sentence: {sentence}")
            print(f"Pronoun: {pronoun}")
            print(f"Candidates: {candidates}")
            resolved = resolve_pronouns(nlp, sentence)
            print(f"Resolved: {resolved}")

            resolved_results.append({
            "original": sentence,
            "resolved": resolved,
            "pronoun": row["pronoun"],
            "candidates": row["candidates"],
            })

        df_resolved = pd.DataFrame(resolved_results)
        df_resolved.to_csv("resolved_dataset.csv", index=False)
        print("\n[Saved resolved dataset to resolved_dataset.csv]")

    
    nlp = load_model()
    print("\nChoose mode:")
    print("1. Manual paragraph input")
    print("2. Run on local dataset (dpr_train.csv)")
    choice = input("> ").strip()
    
    if choice == "1":
        print("\nEnter a paragraph of text:")
        paragraph = input("> ")
        resolved_text = resolve_pronouns(nlp, paragraph)
        print("\n[Original Paragraph]:")
        print(paragraph)
        print("\n[Resolved Paragraph]:")
        print(resolved_text)

    elif choice == "2":
        run_dataset_mode(nlp)
    else:
        print("Invalid selection.")

def run_accuracy():
    # Run accuracy evaluation
    print("\n" + "="*100)
    print("ACCURACY EVALUATION")
    print("="*100)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "evaluate_accuracy.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return result.returncode == 0

def run_topic_comparison():
    # Run LDA topic modeling comparison
    print("\n" + "="*100)
    print("TOPIC MODELING COMPARISON")
    print("="*100)
    print("(This may take 5-10 minutes)")
    
    import subprocess
    result = subprocess.run(
        [sys.executable, "topic_modeling_comparison.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode == 0:
        print("\n[Report saved to: topic_comparison_report.txt]")
    return result.returncode == 0

def main():
    # Main entry point
    while True:
        choice = show_menu()
        
        if choice == "1":
            try:
                run_interactive()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            try:
                run_accuracy()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            try:
                run_topic_comparison()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "4":
            print("\n[Exiting...]")
            break
        
        else:
            print("Invalid choice. Please enter 1-4.")

if __name__ == "__main__":
    main()
