# topic_modeling_comparison.py
# Compare LDA topic models on raw vs pronoun-resolved documents, save report

import spacy
import coreferee
import pandas as pd
import numpy as np
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import warnings
import sys
import io
warnings.filterwarnings('ignore')

# Set UTF-8 encoding for output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def load_model():
    # Load spaCy model with coreference resolution
    print("Loading SpaCy + Coreferee...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    return nlp

def resolve_pronouns(nlp, paragraph):
    # Return paragraph with pronouns replaced by their referents
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

def preprocess_text(text):
    # Preprocess text for LDA: lowercase, tokenize, remove stopwords and short words
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokenize into words
    tokens = word_tokenize(text)
    
    # remove common stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    return tokens

def run_lda(documents, num_topics=8, num_words=20):
    # Run LDA on documents and return top words per topic
    print(f"Preprocessing {len(documents)} documents...")
    processed_docs = [preprocess_text(doc) for doc in documents]
    
    # filter out empty documents
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]
    
    print(f"Creating dictionary and corpus...")
    # build dictionary of unique words
    dictionary = corpora.Dictionary(processed_docs)
    # convert documents to bag-of-words format
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    print(f"Running LDA with {num_topics} topics...")
    # train LDA model
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        passes=5,
        per_word_topics=True,
        minimum_probability=0.0,
        alpha='auto'
    )
    
    print(f"Extracting top {num_words} words per topic...")
    # extract top words for each topic
    topics = {}
    for topic_id in range(num_topics):
        terms = lda_model.show_topic(topic_id, topn=num_words)
        topic_words = [term[0] for term in terms]
        topics[topic_id] = topic_words
    
    return topics, lda_model, corpus, dictionary

def main():
    print("\n" + "="*100)
    print("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents")
    print("="*100)
    
    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv('dpr_train.csv')
    documents = df['sentence'].tolist()
    print(f"Loaded {len(documents)} documents")
    
    # LDA on RAW documents
    print("\n" + "="*100)
    print("PART 1: LDA on RAW DOCUMENTS")
    print("="*100)
    
    raw_topics, raw_model, raw_corpus, raw_dict = run_lda(documents, num_topics=8, num_words=20)
    
    print("\n[RAW DOCUMENT TOPICS - Top 20 Words per Topic]")
    print("-" * 100)
    for topic_id, words in raw_topics.items():
        print(f"\nTopic {topic_id}:")
        print(f"  {', '.join(words)}")
    
    # Pronoun resolution
    print("\n" + "="*100)
    print("PART 2: Running Pronoun Resolution on Documents")
    print("="*100)
    
    nlp = load_model()
    print(f"\nResolving pronouns in {len(documents)} documents...")
    
    # apply pronoun resolution to all documents
    resolved_documents = []
    for i, doc in enumerate(documents):
        resolved_doc = resolve_pronouns(nlp, doc)
        resolved_documents.append(resolved_doc)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(documents)} documents")
    
    print(f"Pronoun resolution complete!")
    
    #LDA on RESOLVED documents
    print("\n" + "="*100)
    print("PART 3: LDA on PRONOUN-RESOLVED DOCUMENTS")
    print("="*100)
    
    resolved_topics, resolved_model, resolved_corpus, resolved_dict = run_lda(
        resolved_documents, num_topics=8, num_words=20
    )
    
    print("\nRESOLVED DOCUMENT TOPICS - Top 20 Words per Topic:")
    print("-" * 100)
    for topic_id, words in resolved_topics.items():
        print(f"\nTopic {topic_id}:")
        print(f"  {', '.join(words)}")
    
    # Comparison 
    print("\n" + "="*100)
    print("PART 4: SIDE-BY-SIDE COMPARISON")
    print("="*100)
    
    for topic_id in range(8):
        raw_words = set(raw_topics[topic_id])
        resolved_words = set(resolved_topics[topic_id])
        
        # calculate overlap between raw and resolved top words
        common = raw_words & resolved_words
        only_raw = raw_words - resolved_words
        only_resolved = resolved_words - raw_words
        
        print(f"\n{'='*100}")
        print(f"TOPIC {topic_id}")
        print(f"{'='*100}")
        print(f"\n[RAW DOCUMENTS (Top 20)]:")
        print(f"  {', '.join(raw_topics[topic_id])}")
        print(f"\n[RESOLVED DOCUMENTS (Top 20)]:")
        print(f"  {', '.join(resolved_topics[topic_id])}")
        print(f"\n[ANALYSIS]:")
        print(f"  Common words: {len(common)}/20 ({100*len(common)/20:.1f}%)")
        print(f"  Words in raw only: {only_raw if only_raw else 'None'}")
        print(f"  Words in resolved only: {only_resolved if only_resolved else 'None'}")
    
    # Summary Statistics 
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # aggregate stats across all topics
    all_raw_words = set()
    all_resolved_words = set()
    
    for topic_id in range(8):
        all_raw_words.update(raw_topics[topic_id])
        all_resolved_words.update(resolved_topics[topic_id])
    
    common_all = all_raw_words & all_resolved_words
    unique_raw = all_raw_words - all_resolved_words
    unique_resolved = all_resolved_words - all_raw_words
    
    print(f"\nTotal unique words (raw): {len(all_raw_words)}")
    print(f"Total unique words (resolved): {len(all_resolved_words)}")
    print(f"Common words across both: {len(common_all)}")
    print(f"Words unique to raw: {len(unique_raw)}")
    print(f"Words unique to resolved: {len(unique_resolved)}")
    print(f"\nOverall similarity: {100*len(common_all)/max(len(all_raw_words), len(all_resolved_words)):.1f}%")
    
    # Save comparison to file
    print("\n" + "="*100)
    print("Saving detailed comparison to 'topic_comparison_report.txt'...")
    
    with open('topic_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents\n")
        f.write("="*100 + "\n\n")
        
        for topic_id in range(8):
            raw_words = set(raw_topics[topic_id])
            resolved_words = set(resolved_topics[topic_id])
            
            common = raw_words & resolved_words
            only_raw = raw_words - resolved_words
            only_resolved = resolved_words - raw_words
            
            f.write(f"\n{'='*100}\n")
            f.write(f"TOPIC {topic_id}\n")
            f.write(f"{'='*100}\n")
            f.write(f"\n[RAW DOCUMENTS (Top 20)]:\n")
            f.write(f"  {', '.join(raw_topics[topic_id])}\n")
            f.write(f"\n[RESOLVED DOCUMENTS (Top 20)]:\n")
            f.write(f"  {', '.join(resolved_topics[topic_id])}\n")
            f.write(f"\n[ANALYSIS]:\n")
            f.write(f"  Common words: {len(common)}/20 ({100*len(common)/20:.1f}%)\n")
            f.write(f"  Words in raw only: {', '.join(only_raw) if only_raw else 'None'}\n")
            f.write(f"  Words in resolved only: {', '.join(only_resolved) if only_resolved else 'None'}\n")
        
        f.write(f"\n\n{'='*100}\n")
        f.write("SUMMARY STATISTICS\n")
        f.write(f"{'='*100}\n\n")
        f.write(f"Total unique words (raw): {len(all_raw_words)}\n")
        f.write(f"Total unique words (resolved): {len(all_resolved_words)}\n")
        f.write(f"Common words across both: {len(common_all)}\n")
        f.write(f"Words unique to raw: {len(unique_raw)}\n")
        f.write(f"Words unique to resolved: {len(unique_resolved)}\n")
        f.write(f"Overall similarity: {100*len(common_all)/max(len(all_raw_words), len(all_resolved_words)):.1f}%\n")
    
    print("Report saved!")
    print("\nAnalysis complete!\n")

if __name__ == "__main__":
    main()
