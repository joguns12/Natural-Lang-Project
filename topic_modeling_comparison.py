import spacy
import coreferee
import pandas as pd
import numpy as np
import re
import nltk
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.optimize import linear_sum_assignment
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def count_name_frequencies(docs, names):
    freq = {name: 0 for name in names}
    for doc in docs:
        doc_lower = doc.lower()
        for name in names:
            freq[name] += len(re.findall(rf'\b{name}\b', doc_lower))
    return freq

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.where(p == 0, 1e-12, p)
    q = np.where(q == 0, 1e-12, q)
    return np.sum(p * np.log(p / q))

def get_topic_dist(model, topic_id, vocab):
    topic_terms = dict(model.get_topic_terms(topic_id, topn=len(vocab)))
    dist = np.array([topic_terms.get(word_id, 1e-12) for word_id in range(len(vocab))], dtype=np.float64)
    dist /= dist.sum()
    return dist

def load_model():
    print("Loading SpaCy + Coreferee...")
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("coreferee")
    return nlp

def resolve_pronouns(nlp, paragraph):
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
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    return tokens

def run_lda(documents, num_topics=8, num_words=20):
    print(f"Preprocessing {len(documents)} documents...")
    processed_docs = [preprocess_text(doc) for doc in documents]
    processed_docs = [doc for doc in processed_docs if len(doc) > 0]
    print(f"Creating dictionary and corpus...")
    dictionary = corpora.Dictionary(processed_docs)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
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
    topics = {}
    for topic_id in range(num_topics):
        terms = lda_model.show_topic(topic_id, topn=num_words)
        topic_words = [term[0] for term in terms]
        topics[topic_id] = topic_words
    return topics, lda_model, corpus, dictionary

def topic_kl_matrix(model_left, dict_left, model_right, dict_right, num_topics=8):
    # Build union vocab
    vocab_l = dict_left.token2id
    vocab_r = dict_right.token2id
    union_vocab = list(set(vocab_l.keys()) | set(vocab_r.keys()))
    union_word2id = {w: i for i, w in enumerate(union_vocab)}
    # Precompute distributions
    dists_l = []
    for t in range(num_topics):
        dist = np.zeros(len(union_vocab))
        terms = dict(model_left.get_topic_terms(t, topn=len(vocab_l)))
        for w, i in union_word2id.items():
            wid = dict_left.token2id.get(w)
            dist[i] = terms.get(wid, 1e-12) if wid is not None else 1e-12
        dist /= dist.sum()
        dists_l.append(dist)
    dists_r = []
    for t in range(num_topics):
        dist = np.zeros(len(union_vocab))
        terms = dict(model_right.get_topic_terms(t, topn=len(vocab_r)))
        for w, i in union_word2id.items():
            wid = dict_right.token2id.get(w)
            dist[i] = terms.get(wid, 1e-12) if wid is not None else 1e-12
        dist /= dist.sum()
        dists_r.append(dist)
    # KL matrix
    M = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(num_topics):
            M[i, j] = kl_divergence(dists_l[i], dists_r[j])
    return M

def main():
    # Visual style for professional charts
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({
        'figure.dpi': 140,
        'savefig.dpi': 200,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'axes.titlesize': 'large',
        'axes.labelsize': 'medium',
        'legend.frameon': False,
    })
    # Configurable seeds for experimentation (used in baseline runs)
    baseline_seeds = [42, 43]  # adjust to experiment
    candidate_names = ["john", "mary", "bill", "bob", "jane", "jack", "jim", "joe", "susan", "jennifer"]
    print("\n" + "="*100)
    print("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents")
    print("="*100)

    print("\nLoading dataset from dpr_train.csv and dpr_test.csv...")
    df_train = pd.read_csv('dpr_train.csv')
    df_test = pd.read_csv('dpr_test.csv')
    documents = df_train['sentence'].tolist() + df_test['sentence'].tolist()
    print(f"Loaded {len(documents)} documents from CSVs")

    print("\nName frequencies in RAW documents:")
    raw_freq = count_name_frequencies(documents, candidate_names)
    for name, count in raw_freq.items():
        print(f"  {name}: {count}")

    print("\n" + "="*100)
    print("PART 1: LDA on RAW DOCUMENTS")
    print("="*100)
    raw_topics, raw_model, raw_corpus, raw_dict = run_lda(documents, num_topics=8, num_words=20)
    print("\n[RAW DOCUMENT TOPICS - Top 20 Words per Topic]")
    print("-" * 100)
    for topic_id, words in raw_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

    print("\n" + "="*100)
    print("PART 2: Running Pronoun Resolution on Documents")
    print("="*100)
    nlp = load_model()
    print(f"\nResolving pronouns in {len(documents)} documents...")
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
    # Compute per-name uplift stats
    uplift_rows = []
    for name in candidate_names:
        raw_c = raw_freq.get(name, 0)
        res_c = resolved_freq.get(name, 0)
        delta = res_c - raw_c
        pct = (delta / raw_c * 100.0) if raw_c > 0 else (100.0 if res_c > 0 else 0.0)
        uplift_rows.append((name, raw_c, res_c, delta, pct))
    # Show top-3 names by percent increase
    top_uplift = sorted(uplift_rows, key=lambda x: x[4], reverse=True)[:3]
    print("\nTop-3 name uplifts (percent increase):")
    for name, raw_c, res_c, delta, pct in top_uplift:
        print(f"  {name}: raw={raw_c}, resolved={res_c}, delta={delta}, +{pct:.1f}%")

    print("\n" + "="*100)
    print("PART 3: LDA on PRONOUN-RESOLVED DOCUMENTS")
    print("="*100)
    resolved_topics, resolved_model, resolved_corpus, resolved_dict = run_lda(resolved_documents, num_topics=8, num_words=20)
    print("\nRESOLVED DOCUMENT TOPICS - Top 20 Words per Topic:")
    print("-" * 100)
    for topic_id, words in resolved_topics.items():
        print(f"Topic {topic_id}: {', '.join(words)}")

    print("\n" + "="*100)
    print("PART 4: TOPIC MATCHING & KL DIVERGENCE")
    print("="*100)
    # KL-based one-to-one matching between RAW and RESOLVED
    M_rr = topic_kl_matrix(raw_model, raw_dict, resolved_model, resolved_dict, num_topics=8)
    row_ind, col_ind = linear_sum_assignment(M_rr)  # minimize total KL
    matches = [(int(r), int(c), float(M_rr[r, c])) for r, c in zip(row_ind, col_ind)]
    used_raw = set(row_ind.tolist()) if hasattr(row_ind, 'tolist') else set(row_ind)
    used_res = set(col_ind.tolist()) if hasattr(col_ind, 'tolist') else set(col_ind)
    unmatched_raw = set(range(8)) - used_raw
    unmatched_resolved = set(range(8)) - used_res
    print("Matched topic pairs (RAW -> RESOLVED) by KL:")
    for r, c, v in matches:
        print(f"  Raw {r} -> Resolved {c} (KL={v:.4f})")
    print(f"Unmatched raw topics: {sorted(list(unmatched_raw)) if unmatched_raw else 'None'}")
    print(f"Unmatched resolved topics: {sorted(list(unmatched_resolved)) if unmatched_resolved else 'None'}")

    print("\nKL divergence for matched topic pairs:")
    kl_values = [v for _, _, v in matches]
    kl_pairs = matches
    print(f"\nAverage KL divergence (RAW vs RESOLVED): {np.mean(kl_values):.4f}")
    print(f"KL divergence values: {[round(k, 4) for k in kl_values]}")

    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
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
    # Additional, more informative similarity metrics
    # 1) Dictionary-level Jaccard over full vocabularies
    raw_vocab = set(raw_dict.token2id.keys())
    resolved_vocab = set(resolved_dict.token2id.keys())
    jaccard_dict = (len(raw_vocab & resolved_vocab) / len(raw_vocab | resolved_vocab)) if (raw_vocab or resolved_vocab) else 0.0

    # 2) Mean per-topic Jaccard over top-20 words after KL-based matching
    jaccard_per_topic = []
    for r, c, _ in matches:
        rw = set(raw_topics[r])
        zw = set(resolved_topics[c])
        union = rw | zw
        inter = rw & zw
        jaccard_per_topic.append((len(inter) / len(union)) if union else 0.0)
    mean_jaccard_top20 = float(np.mean(jaccard_per_topic)) if jaccard_per_topic else 0.0

    # 3) Cosine similarity, Symmetric KL, and Jensen–Shannon distance over matched topic distributions
    # Build union vocab once (same as topic_kl_matrix) and precompute distributions
    vocab_l = raw_dict.token2id
    vocab_r = resolved_dict.token2id
    union_vocab = list(set(vocab_l.keys()) | set(vocab_r.keys()))
    union_word2id = {w: i for i, w in enumerate(union_vocab)}
    def topic_dist(model, dict_, t):
        dist = np.zeros(len(union_vocab))
        terms = dict(model.get_topic_terms(t, topn=len(dict_.token2id)))
        for w, i in union_word2id.items():
            wid = dict_.token2id.get(w)
            dist[i] = terms.get(wid, 1e-12) if wid is not None else 1e-12
        s = dist.sum()
        return dist / s if s > 0 else dist

    def cosine(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    def sym_kl(a, b):
        return 0.5 * (kl_divergence(a, b) + kl_divergence(b, a))

    def js_distance(a, b):
        m = 0.5 * (a + b)
        js_div = 0.5 * (kl_divergence(a, m) + kl_divergence(b, m))
        return float(np.sqrt(js_div))

    cos_vals, skl_vals, js_vals = [], [], []
    for r, c, _ in matches:
        pr = topic_dist(raw_model, raw_dict, r)
        qr = topic_dist(resolved_model, resolved_dict, c)
        cos_vals.append(cosine(pr, qr))
        skl_vals.append(sym_kl(pr, qr))
        js_vals.append(js_distance(pr, qr))

    mean_cosine = float(np.mean(cos_vals)) if cos_vals else 0.0
    mean_symkl = float(np.mean(skl_vals)) if skl_vals else 0.0
    mean_js = float(np.mean(js_vals)) if js_vals else 0.0

    print(f"\nOverall similarity (top-20 overlap ratio): {100*len(common_all)/max(len(all_raw_words), len(all_resolved_words)):.1f}%")
    print(f"Dictionary Jaccard: {jaccard_dict:.3f}")
    print(f"Mean matched-topic Jaccard (top-20): {mean_jaccard_top20:.3f}")
    print(f"Mean cosine similarity (matched topics): {mean_cosine:.3f}")
    print(f"Mean symmetric KL (matched topics): {mean_symkl:.3f}")
    print(f"Mean JS distance (matched topics): {mean_js:.3f}")

    print("\n" + "="*100)
    print("Saving detailed comparison to 'topic_comparison_report.txt'...")
    with open('topic_comparison_report.txt', 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents\n")
        f.write("="*100 + "\n\n")
        # Name uplift section
        f.write("Name Uplift (Raw vs Pronoun-Resolved)\n")
        f.write("-"*100 + "\n")
        f.write("name, raw_count, resolved_count, delta, percent_increase\n")
        for name, raw_c, res_c, delta, pct in uplift_rows:
            f.write(f"{name}, {raw_c}, {res_c}, {delta}, {pct:.1f}%\n")
        f.write("\n")
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
        # Write enhanced metrics
        f.write(f"Overall similarity (top-20 overlap ratio): {100*len(common_all)/max(len(all_raw_words), len(all_resolved_words)):.1f}%\n")
        f.write(f"Dictionary Jaccard: {jaccard_dict:.3f}\n")
        f.write(f"Mean matched-topic Jaccard (top-20): {mean_jaccard_top20:.3f}\n")
        f.write(f"Mean cosine similarity (matched topics): {mean_cosine:.3f}\n")
        f.write(f"Mean symmetric KL (matched topics): {mean_symkl:.3f}\n")
        f.write(f"Mean JS distance (matched topics): {mean_js:.3f}\n")
    print("Report saved!")

    # Baseline: Two-run topic comparison on RAW documents
    print("\n" + "="*100)
    print("BASELINE: Two-Run Topic Comparison (RAW Documents)")
    print("="*100)
    # Run LDA twice with configurable seeds
    # First run
    raw_proc = [preprocess_text(doc) for doc in documents]
    raw_proc = [d for d in raw_proc if len(d) > 0]
    dict_a = corpora.Dictionary(raw_proc)
    corpus_a = [dict_a.doc2bow(d) for d in raw_proc]
    model_a = models.LdaModel(corpus=corpus_a, id2word=dict_a, num_topics=8, random_state=baseline_seeds[0], passes=5, per_word_topics=True, minimum_probability=0.0, alpha='auto')
    topics_a = {tid: [w for w, _ in model_a.show_topic(tid, topn=20)] for tid in range(8)}
    # Second run
    dict_b = dict_a  # keep same dictionary for fair comparison
    corpus_b = corpus_a
    model_b = models.LdaModel(corpus=corpus_b, id2word=dict_b, num_topics=8, random_state=baseline_seeds[1], passes=5, per_word_topics=True, minimum_probability=0.0, alpha='auto')
    topics_b = {tid: [w for w, _ in model_b.show_topic(tid, topn=20)] for tid in range(8)}

    # KL-based matching between Run A and Run B
    M_ab = topic_kl_matrix(model_a, dict_a, model_b, dict_b, num_topics=8)
    r_ind, c_ind = linear_sum_assignment(M_ab)
    matches_ab = [(int(r), int(c), float(M_ab[r, c])) for r, c in zip(r_ind, c_ind)]
    unmatched_a = set(range(8)) - set(r_ind)
    unmatched_b = set(range(8)) - set(c_ind)

    print("Matched topic pairs (run A -> run B):")
    for a_id, b_id, ov in matches_ab:
        print(f"  A{a_id} -> B{b_id} (overlap={ov})")
    print(f"Unmatched A topics: {sorted(list(unmatched_a)) if unmatched_a else 'None'}")
    print(f"Unmatched B topics: {sorted(list(unmatched_b)) if unmatched_b else 'None'}")

    # KL values for up to 5 matches (we already have them in matches_ab)
    kl_ab_values = [(a, b, v) for a, b, v in matches_ab[:5]]
    for a_id, b_id, klv in kl_ab_values:
        print(f"  KL(A{a_id}, B{b_id}) = {klv:.4f}")

    # Append baseline comparison to report
    with open('topic_comparison_report.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + "="*100 + "\n")
        f.write("BASELINE: Two-Run Topic Comparison (RAW Documents)\n")
        f.write("="*100 + "\n\n")
        for a_id, b_id, klv in matches_ab:
            f.write(f"Match A{a_id} -> B{b_id} (KL={klv:.4f})\n")
        f.write(f"Unmatched A topics: {sorted(list(unmatched_a)) if unmatched_a else 'None'}\n")
        f.write(f"Unmatched B topics: {sorted(list(unmatched_b)) if unmatched_b else 'None'}\n")
        f.write(f"Summary: matched={len(matches_ab)}, unmatchedA={len(unmatched_a)}, unmatchedB={len(unmatched_b)}\n")
        f.write("KL for up to 5 matches:\n")
        for a_id, b_id, klv in kl_ab_values:
            f.write(f"  KL(A{a_id}, B{b_id}) = {klv:.4f}\n")
        if not kl_ab_values:
            f.write("  No matches available to compute KL.\n")
    # Export CSVs for plotting/analysis
    with open('baseline_kl.csv', 'w', newline='', encoding='utf-8') as cf:
        w = csv.writer(cf)
        w.writerow(['A_topic', 'B_topic', 'KL'])
        for a_id, b_id, klv in matches_ab:
            w.writerow([a_id, b_id, klv])
    with open('raw_resolved_kl.csv', 'w', newline='', encoding='utf-8') as cf:
        w = csv.writer(cf)
        w.writerow(['Raw_topic', 'Resolved_topic', 'KL'])
        for r, c, v in kl_pairs:
            w.writerow([r, c, v])
    print("Baseline two-run comparison appended to report.")

    # Append side-by-side summary comparing baseline vs raw-resolved KLs
    try:
        # Load CSVs
        baseline_vals = []
        with open('baseline_kl.csv', 'r', encoding='utf-8') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                try:
                    baseline_vals.append(float(row['KL']))
                except Exception:
                    pass
        rr_vals = []
        with open('raw_resolved_kl.csv', 'r', encoding='utf-8') as cf:
            reader = csv.DictReader(cf)
            for row in reader:
                try:
                    rr_vals.append(float(row['KL']))
                except Exception:
                    pass
        def stats(vals):
            if not vals:
                return {'count': 0, 'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            arr = np.array(vals, dtype=np.float64)
            return {
                'count': int(arr.size),
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'std': float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
            }
        bstats = stats(baseline_vals)
        rrstats = stats(rr_vals)
        # Write summary to report
        with open('topic_comparison_report.txt', 'a', encoding='utf-8') as f:
            f.write("\n" + "="*100 + "\n")
            f.write("BASELINE vs RAW→RESOLVED KL SUMMARY\n")
            f.write("="*100 + "\n")
            f.write("Metric, Baseline KL, Raw→Resolved KL\n")
            f.write(f"Count, {bstats['count']}, {rrstats['count']}\n")
            f.write(f"Mean, {bstats['mean']:.4f}, {rrstats['mean']:.4f}\n")
            f.write(f"Median, {bstats['median']:.4f}, {rrstats['median']:.4f}\n")
            f.write(f"StdDev, {bstats['std']:.4f}, {rrstats['std']:.4f}\n")
            f.write(f"Min, {bstats['min']:.4f}, {rrstats['min']:.4f}\n")
            f.write(f"Max, {bstats['max']:.4f}, {rrstats['max']:.4f}\n")
            # Simple interpretation line
            delta_mean = rrstats['mean'] - bstats['mean']
            f.write(f"\nInterpretation: Raw→Resolved mean KL is {delta_mean:+.4f} higher than baseline, indicating transformation impact beyond seed variability.\n")
        print("Baseline vs Raw→Resolved KL summary appended to report.")
        # Generate histogram plot (professional style)
        try:
            plt.figure(figsize=(8, 5))
            bins = 10
            # Histograms
            plt.hist(baseline_vals, bins=bins, alpha=0.6, label='Baseline KL', color='#1f77b4')
            plt.hist(rr_vals, bins=bins, alpha=0.6, label='Raw→Resolved KL', color='#ff7f0e')
            # Mean markers
            b_mean = np.mean(baseline_vals) if baseline_vals else 0.0
            rr_mean = np.mean(rr_vals) if rr_vals else 0.0
            plt.axvline(b_mean, color='#1f77b4', linestyle='--', linewidth=2, label=f'Baseline mean={b_mean:.3f}')
            plt.axvline(rr_mean, color='#ff7f0e', linestyle='--', linewidth=2, label=f'Raw→Resolved mean={rr_mean:.3f}')
            plt.title('KL Divergence Distribution: Baseline vs Raw→Resolved')
            plt.xlabel('KL divergence')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig('kl_histogram.png')
            plt.close()
            print("Saved KL histogram to kl_histogram.png")
        except Exception as pe:
            print(f"Failed to save histogram: {pe}")
        # Additional plots for other metrics (raw→resolved only) using KDE density plots
        try:
            def kde_plot(values, title, outfile, xlabel, color, xlim=None):
                data = np.array(values, dtype=float)
                if data.size == 0:
                    return
                plt.figure(figsize=(8, 5))
                try:
                    kde = gaussian_kde(data)
                    xmin = float(np.min(data)) if xlim is None else xlim[0]
                    xmax = float(np.max(data)) if xlim is None else xlim[1]
                    # Expand a bit for nicer tails
                    rng = xmax - xmin
                    xmin -= 0.05 * rng
                    xmax += 0.05 * rng
                    xs = np.linspace(xmin, xmax, 256)
                    plt.plot(xs, kde(xs), color=color, linewidth=2, label='Density')
                except Exception:
                    # Fallback: smoothed histogram line
                    counts, bin_edges = np.histogram(data, bins=10, density=True)
                    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    plt.plot(centers, counts, color=color, linewidth=2, label='Density (binned)')
                mean_val = float(np.mean(data))
                plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel('Density')
                plt.legend()
                plt.tight_layout()
                plt.savefig(outfile)
                plt.close()
                print(f'Saved {title} to {outfile}')

            if cos_vals:
                kde_plot(cos_vals, 'Cosine Similarity Density (Matched Topics)', 'cosine_density.png', 'Cosine Similarity', '#2ca02c', xlim=(0.0, 1.0))
            if skl_vals:
                kde_plot(skl_vals, 'Symmetric KL Density (Matched Topics)', 'symkl_density.png', 'Symmetric KL', '#9467bd')
            if js_vals:
                kde_plot(js_vals, 'Jensen–Shannon Distance Density (Matched Topics)', 'js_density.png', 'JS Distance', '#8c564b', xlim=(0.0, 1.0))
            if jaccard_per_topic:
                kde_plot(jaccard_per_topic, 'Top-20 Jaccard Density (Matched Topics)', 'jaccard_density.png', 'Jaccard (Top-20)', '#e377c2', xlim=(0.0, 1.0))
        except Exception as me:
            print(f"Failed to save metric density plots: {me}")
    except Exception as e:
        print(f"Failed to append KL summary: {e}")


if __name__ == "__main__":
    main()