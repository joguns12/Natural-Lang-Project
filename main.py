# main.py
# Pronoun Resolution Wrapper — Joshua Ogunseinde
# Reads a paragraph, performs pronoun resolution, and outputs resolved text.

import spacy
import coreferee

def load_model():
    """Load spaCy model with coreference resolution."""
    print("Loading SpaCy + Coreferee...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    return nlp

def resolve_pronouns(nlp, paragraph):
    #Return paragraph with pronouns replaced by their referents."""
    doc = nlp(paragraph)
    if doc._.coref_chains:
        resolved = doc._.coref_chains.resolve(doc)
        return resolved
    else:
        return paragraph  # No pronouns found


def main():
    nlp = load_model()
    print("\nEnter a paragraph of text:")
    paragraph = input("> ")

    resolved_text = resolve_pronouns(nlp, paragraph)

    print("\n Original Paragraph")
    print(paragraph)
    print("\n Resolved Paragraph")
    print(resolved_text)

if __name__ == "__main__":
    main()
