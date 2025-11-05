# main.py
# Pronoun Resolution Wrapper 
# Reads a paragraph or dataset sentence, performs pronoun resolution, and outputs resolved text.
import spacy
import coreferee
import pandas as pd
import random

def load_model():
    #Load spaCy model with coreference resolution.
    print("Loading SpaCy + Coreferee...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    return nlp

def resolve_pronouns(nlp, paragraph):
    #Return paragraph with pronouns replaced by their referents.
    doc = nlp(paragraph)
    if doc._.coref_chains:
        resolved_text = paragraph
        for chain in doc._.coref_chains:
            main_idx = chain.most_specific_mention_index
            main_mention = chain.mentions[main_idx]
            main_text = doc[main_mention[0]: main_mention[-1] + 1].text

            for mention in chain.mentions:
                if mention == main_mention:
                    continue
                mention_text = doc[mention[0]: mention[-1] + 1].text
                resolved_text = resolved_text.replace(mention_text, main_text)
        return resolved_text
    else:
        return paragraph

def run_dataset_mode(nlp, file_path="dpr_train.csv", sample_size=5):
   #Run the resolver on sample sentences from the local DPR dataset
    print(f"\nLoading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    examples = df.sample(sample_size, random_state=42)

    for _, row in examples.iterrows():
        sentence = row["sentence"]
        pronoun = row["pronoun"]
        candidates = row["candidates"]
        print("\n--------------------------------------------")
        print(f"Sentence: {sentence}")
        print(f"Pronoun: {pronoun}")
        print(f"Candidates: {candidates}")
        resolved = resolve_pronouns(nlp, sentence)
        print(f"Resolved: {resolved}")

def main():
    nlp = load_model()
    print("\nChoose mode:")
    print("1. Manual paragraph input")
    print("2. Run on local dataset (dpr_train.csv)")
    choice = input("> ").strip()

    if choice == "1":
        print("\nEnter a paragraph of text:")
        paragraph = input("> ")
        resolved_text = resolve_pronouns(nlp, paragraph)
        print("\nOriginal Paragraph:")
        print(paragraph)
        print("\nResolved Paragraph:")
        print(resolved_text)
    elif choice == "2":
        run_dataset_mode(nlp)
    else:
        print("Invalid selection. Please run again and choose 1 or 2.")

if __name__ == "__main__":
    main()
