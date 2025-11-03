# main.py
# Pronoun Resolution Wrapper — Joshua Ogunseinde
# Reads a paragraph, performs pronoun resolution, and outputs resolved text.

import spacy
import coreferee

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
            # coreferee Chain objects provide `.mentions` (lists of token indices)
            # and `.most_specific_mention_index` (index into `.mentions`).
            # Build main mention span from the token indices.
            main_idx = chain.most_specific_mention_index
            main_mention = chain.mentions[main_idx]
            # `main_mention` is a list of token indices (inclusive). Use the
            # first and last token index to slice the spaCy doc.
            main_text = doc[main_mention[0]: main_mention[-1] + 1].text

            # Replace pronouns and secondary mentions
            for mention in chain.mentions:
                if mention == main_mention:
                    continue
                mention_text = doc[mention[0]: mention[-1] + 1].text
                resolved_text = resolved_text.replace(mention_text, main_text)
        return resolved_text
    else:
        return paragraph  # No pronouns found

def main():
    nlp = load_model()
    print("\nEnter a paragraph of text:")
    paragraph = input("> ")

    resolved_text = resolve_pronouns(nlp, paragraph)

    print("\n--- Original Paragraph ---")
    print(paragraph)
    print("\n--- Resolved Paragraph ---")
    print(resolved_text)

if __name__ == "__main__":
    main()
    