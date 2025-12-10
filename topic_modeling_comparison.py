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
           random_state=np.random.randint(0, 100000),
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
        # Baseline: Run LDA multiple times on raw data and compute average KL divergence between runs
        print("\n" + "="*100)
        print("BASELINE: LDA Variability on RAW Data")
        print("="*100)
        baseline_runs = 5
        baseline_kls = []
        baseline_topics = []
        baseline_models = []
        processed_docs = [preprocess_text(doc) for doc in documents]
        processed_docs = [doc for doc in processed_docs if len(doc) > 0]
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        for i in range(baseline_runs):
            lda_model = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=8,
                random_state=42 + i,
                passes=5,
                per_word_topics=True,
                minimum_probability=0.0,
                alpha='auto'
            )
            topics = {}
            for topic_id in range(8):
                terms = lda_model.show_topic(topic_id, topn=20)
                topic_words = [term[0] for term in terms]
                topics[topic_id] = topic_words
            baseline_topics.append(topics)
            baseline_models.append(lda_model)

        # Compute KL divergence between each pair of baseline runs
        def get_topic_dist(model, topic_id, vocab):
            topic_terms = dict(model.get_topic_terms(topic_id, topn=len(vocab)))
            dist = np.array([topic_terms.get(word_id, 1e-12) for word_id in range(len(vocab))], dtype=np.float64)
            dist /= dist.sum()
            return dist

        for i in range(baseline_runs):
            for j in range(i+1, baseline_runs):
                # Build union vocab
                vocab_i = baseline_models[i].id2word.token2id
                vocab_j = baseline_models[j].id2word.token2id
                union_vocab = list(set(vocab_i.keys()) | set(vocab_j.keys()))
                union_word2id = {word: idx for idx, word in enumerate(union_vocab)}
                for topic_id in range(8):
                    dist_i = np.zeros(len(union_vocab))
                    dist_j = np.zeros(len(union_vocab))
                    terms_i = dict(baseline_models[i].get_topic_terms(topic_id, topn=len(vocab_i)))
                    terms_j = dict(baseline_models[j].get_topic_terms(topic_id, topn=len(vocab_j)))
                    for word, idx in union_word2id.items():
                        id_i = vocab_i.get(word)
                        id_j = vocab_j.get(word)
                        dist_i[idx] = terms_i.get(id_i, 1e-12) if id_i is not None else 1e-12
                        dist_j[idx] = terms_j.get(id_j, 1e-12) if id_j is not None else 1e-12
                    dist_i /= dist_i.sum()
                    dist_j /= dist_j.sum()
                    kl = kl_divergence(dist_i, dist_j)
                    baseline_kls.append(kl)

        # ...existing code...
        # Save main comparison to file (overwrite)
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
                f.write(f"\n--- Topic {topic_id} ---\n")
                f.write(f"[RAW DOCUMENTS (Top 20)]:\n")
                f.write(f"  {', '.join(raw_topics[topic_id])}\n")
                f.write(f"[RESOLVED DOCUMENTS (Top 20)]:\n")
                f.write(f"  {', '.join(resolved_topics[topic_id])}\n")
                f.write(f"[ANALYSIS]:\n")
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

        # Append baseline KL divergence results
        with open('topic_comparison_report.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n\n{'='*100}\n")
            f.write("BASELINE KL DIVERGENCE\n")
            f.write(f"{'='*100}\n\n")
            f.write(f"Baseline average KL divergence (raw vs. raw): {np.mean(baseline_kls):.4f}\n")
            f.write(f"Individual baseline KL divergence values:\n")
            for idx, kl in enumerate(baseline_kls):
                f.write(f"  Pair {idx+1}: KL = {kl:.4f}\n")
        print("Report saved!")
        print("\nAnalysis complete!\n")
    # End of baseline block, continue with main analysis
print("\n" + "="*100)
print("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents")
print("="*100)

# Load data from dpr_train.csv and dpr_testy.csv
print("\nLoading dataset from dpr_train.csv and dpr_test.csv...")
df_train = pd.read_csv('dpr_train.csv')
df_test = pd.read_csv('dpr_test.csv')
documents = df_train['sentence'].tolist() + df_test['sentence'].tolist()
print(f"Loaded {len(documents)} documents from CSVs")

# Candidate names to check frequency for
candidate_names = ["john", "mary", "bill", "bob", "jane", "jack", "jim", "joe", "susan", "jennifer"]

def count_name_frequencies(docs, names):
    freq = {name: 0 for name in names}
    for doc in docs:
        doc_lower = doc.lower()
        for name in names:
            # Count whole word matches only
            freq[name] += len(re.findall(rf'\b{name}\b', doc_lower))
    return freq

print("\nName frequencies in RAW documents:")
raw_freq = count_name_frequencies(documents, candidate_names)
for name, count in raw_freq.items():
    print(f"  {name}: {count}")
    
    # LDA on RAW documents
    print("\n" + "="*100)
    print("PART 1: LDA on RAW DOCUMENTS")
    print("="*100)
    
    raw_topics, raw_model, raw_corpus, raw_dict = run_lda(documents, num_topics=8, num_words=20)
    
    # Print raw topics summary only once
    print("\n[RAW DOCUMENT TOPICS - Top 20 Words per Topic]")
    print("-" * 100)
    for topic_id, words in raw_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")
    
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

    print("\nName frequencies in PRONOUN-RESOLVED documents:")
    resolved_freq = count_name_frequencies(resolved_documents, candidate_names)
    for name, count in resolved_freq.items():
        print(f"  {name}: {count}")
    
    #LDA on RESOLVED documents
    print("\n" + "="*100)
    print("PART 3: LDA on PRONOUN-RESOLVED DOCUMENTS")
    print("="*100)
    
    resolved_topics, resolved_model, resolved_corpus, resolved_dict = run_lda(
        resolved_documents, num_topics=8, num_words=20
    )
    
    # Print resolved topics summary only once
    print("\nRESOLVED DOCUMENT TOPICS - Top 20 Words per Topic:")
    print("-" * 100)
    for topic_id, words in resolved_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")
    

    # Topic matching and KL divergence
    print("\n" + "="*100)
    print("PART 4: TOPIC MATCHING & KL DIVERGENCE")
    print("="*100)

    def kl_divergence(p, q):
        # p, q: numpy arrays of probabilities
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        # Avoid division by zero
        p = np.where(p == 0, 1e-12, p)
        q = np.where(q == 0, 1e-12, q)
        return np.sum(p * np.log(p / q))

    # Get word distributions for all topics
    def get_topic_dist(model, topic_id, vocab):
        # Returns probability vector for all vocab words for a topic
        topic_terms = dict(model.get_topic_terms(topic_id, topn=len(vocab)))
        dist = np.array([topic_terms.get(word_id, 1e-12) for word_id in range(len(vocab))], dtype=np.float64)
        dist /= dist.sum()
        return dist

    # Match topics by maximum word overlap
    matches = []
    unmatched_raw = set(range(8))
    unmatched_resolved = set(range(8))
    # ...existing code...

    num_unmatched_raw = len(unmatched_raw)
    num_unmatched_resolved = len(unmatched_resolved)
    print(f"Unmatched raw topics: {unmatched_raw if unmatched_raw else 'None'}")
    print(f"Unmatched resolved topics: {unmatched_resolved if unmatched_resolved else 'None'}")
    print(f"Number of unmatched raw topics: {num_unmatched_raw}")
    print(f"Number of unmatched resolved topics: {num_unmatched_resolved}")

    # KL divergence for matched pairs
    print("\nKL divergence for matched topic pairs:")
    kl_values = []
    kl_pairs = []
    raw_vocab = raw_dict.token2id
    resolved_vocab = resolved_dict.token2id
    # Build union vocab
    union_vocab = list(set(raw_vocab.keys()) | set(resolved_vocab.keys()))
    union_word2id = {word: i for i, word in enumerate(union_vocab)}

    for raw_id, res_id, _ in matches:
        # Get distributions for union vocab
        raw_dist = np.zeros(len(union_vocab))
        res_dist = np.zeros(len(union_vocab))
        raw_terms = dict(raw_model.get_topic_terms(raw_id, topn=len(raw_dict)))
        res_terms = dict(resolved_model.get_topic_terms(res_id, topn=len(resolved_dict)))
        for word, idx in union_word2id.items():
            raw_id_in_dict = raw_dict.token2id.get(word)
            res_id_in_dict = resolved_dict.token2id.get(word)
            if raw_id_in_dict is not None:
                raw_dist[idx] = raw_terms.get(raw_id_in_dict, 1e-12)
            else:
                raw_dist[idx] = 1e-12
            if res_id_in_dict is not None:
                res_dist[idx] = res_terms.get(res_id_in_dict, 1e-12)
            else:
                res_dist[idx] = 1e-12
        raw_dist /= raw_dist.sum()
        res_dist /= res_dist.sum()
        kl = kl_divergence(raw_dist, res_dist)
        kl_values.append(kl)
        kl_pairs.append((raw_id, res_id, kl))
        print(f"  Raw topic {raw_id} <-> Resolved topic {res_id}: KL = {kl:.4f}")

    print(f"\nAverage KL divergence: {np.mean(kl_values):.4f}")
    print(f"KL divergence values: {[round(k, 4) for k in kl_values]}")
    
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
            f.write(f"\n--- Topic {topic_id} ---\n")
            f.write(f"[RAW DOCUMENTS (Top 20)]:\n")
            f.write(f"  {', '.join(raw_topics[topic_id])}\n")
            f.write(f"[RESOLVED DOCUMENTS (Top 20)]:\n")
            f.write(f"  {', '.join(resolved_topics[topic_id])}\n")
            f.write(f"[ANALYSIS]:\n")
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
