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

def run_lda(documents, num_topics=8, num_words=20, random_state=42):
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
        random_state=random_state,
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
    # Configurable seeds for repeated runs
    num_runs = 5
    baseline_seeds = [42, 43, 44, 45, 46][:num_runs]
    candidate_names = ["john", "mary", "bill", "bob", "jane", "jack", "jim", "joe", "susan", "jennifer"]

    print("\n" + "="*100)
    print("TOPIC MODELING COMPARISON: Raw vs. Pronoun-Resolved Documents (5x runs)")
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
    print("PART 1: Running Pronoun Resolution on Documents")
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

    # Run LDA 5 times for raw and resolved, collect KL divergences
    print("\n" + "="*100)
    print("PART 2: Repeated LDA and KL Divergence Computation")
    print("="*100)
    num_topics = 8
    num_words = 20
    
    # Store models for all-pairs comparisons
    raw_models = []
    raw_dicts = []
    resolved_models = []
    resolved_dicts = []
    
    for i, seed in enumerate(baseline_seeds):
        print(f"\nRun {i+1} / {num_runs} (seed={seed}) for RAW...")
        raw_topics, raw_model, raw_corpus, raw_dict = run_lda(documents, num_topics=num_topics, num_words=num_words, random_state=seed)
        raw_models.append(raw_model)
        raw_dicts.append(raw_dict)
        
        print(f"Run {i+1} / {num_runs} (seed={seed}) for RESOLVED...")
        resolved_topics, resolved_model, resolved_corpus, resolved_dict = run_lda(resolved_documents, num_topics=num_topics, num_words=num_words, random_state=seed)
        resolved_models.append(resolved_model)
        resolved_dicts.append(resolved_dict)
    
    # Compute all pairwise KL divergences for fair comparison
    print("\n" + "="*100)
    print("Computing all pairwise KL divergences...")
    print("="*100)
    
    # 1. RAW vs RESOLVED (different seeds - fair comparison with consecutive)
    kl_raw_vs_resolved = []
    for i in range(num_runs):
        for j in range(num_runs):
            if i != j:  # Don't compare same seed
                M = topic_kl_matrix(raw_models[i], raw_dicts[i], resolved_models[j], resolved_dicts[j], num_topics=num_topics)
                row_ind, col_ind = linear_sum_assignment(M)
                kl_vals = [float(M[r, c]) for r, c in zip(row_ind, col_ind)]
                kl_raw_vs_resolved.append(np.mean(kl_vals))
    
    # 2. RAW vs RAW (different seeds)
    kl_raw_vs_raw = []
    for i in range(num_runs):
        for j in range(i+1, num_runs):  # Only upper triangle to avoid duplicates
            M = topic_kl_matrix(raw_models[i], raw_dicts[i], raw_models[j], raw_dicts[j], num_topics=num_topics)
            row_ind, col_ind = linear_sum_assignment(M)
            kl_vals = [float(M[r, c]) for r, c in zip(row_ind, col_ind)]
            kl_raw_vs_raw.append(np.mean(kl_vals))
    
    # 3. RESOLVED vs RESOLVED (different seeds)
    kl_resolved_vs_resolved = []
    for i in range(num_runs):
        for j in range(i+1, num_runs):  # Only upper triangle to avoid duplicates
            M = topic_kl_matrix(resolved_models[i], resolved_dicts[i], resolved_models[j], resolved_dicts[j], num_topics=num_topics)
            row_ind, col_ind = linear_sum_assignment(M)
            kl_vals = [float(M[r, c]) for r, c in zip(row_ind, col_ind)]
            kl_resolved_vs_resolved.append(np.mean(kl_vals))
    
    # Store matches from first RAW vs RESOLVED comparison for CSV export
    M_export = topic_kl_matrix(raw_models[0], raw_dicts[0], resolved_models[1], resolved_dicts[1], num_topics=num_topics)
    row_ind, col_ind = linear_sum_assignment(M_export)
    kl_pairs = [(int(r), int(c), float(M_export[r, c])) for r, c in zip(row_ind, col_ind)]

    print("\n" + "="*100)
    print("RESULTS: KL Divergence Across All Pairwise Comparisons")
    print("="*100)
    
    mean_raw_vs_resolved = np.mean(kl_raw_vs_resolved)
    mean_raw_vs_raw = np.mean(kl_raw_vs_raw)
    mean_resolved_vs_resolved = np.mean(kl_resolved_vs_resolved)
    
    print(f"\n1. RAW vs RESOLVED (different seeds, {len(kl_raw_vs_resolved)} pairs):")
    print(f"   Mean: {mean_raw_vs_resolved:.4f}")
    print(f"   Std:  {np.std(kl_raw_vs_resolved):.4f}")
    print(f"   Range: [{min(kl_raw_vs_resolved):.4f}, {max(kl_raw_vs_resolved):.4f}]")
    
    print(f"\n2. RAW vs RAW (different seeds, {len(kl_raw_vs_raw)} pairs):")
    print(f"   Mean: {mean_raw_vs_raw:.4f}")
    print(f"   Std:  {np.std(kl_raw_vs_raw):.4f}")
    print(f"   Range: [{min(kl_raw_vs_raw):.4f}, {max(kl_raw_vs_raw):.4f}]")
    
    print(f"\n3. RESOLVED vs RESOLVED (different seeds, {len(kl_resolved_vs_resolved)} pairs):")
    print(f"   Mean: {mean_resolved_vs_resolved:.4f}")
    print(f"   Std:  {np.std(kl_resolved_vs_resolved):.4f}")
    print(f"   Range: [{min(kl_resolved_vs_resolved):.4f}, {max(kl_resolved_vs_resolved):.4f}]")
    
    # Effect size calculation
    effect_above_raw_noise = mean_raw_vs_resolved - mean_raw_vs_raw
    effect_above_resolved_noise = mean_raw_vs_resolved - mean_resolved_vs_resolved
    
    print(f"\n" + "="*100)
    print("INTERPRETATION:")
    print("="*100)
    print(f"Pronoun effect above RAW seed noise:      {effect_above_raw_noise:+.4f} ({effect_above_raw_noise/mean_raw_vs_raw*100:+.1f}%)")
    print(f"Pronoun effect above RESOLVED seed noise: {effect_above_resolved_noise:+.4f} ({effect_above_resolved_noise/mean_resolved_vs_resolved*100:+.1f}%)")
    if mean_resolved_vs_resolved < mean_raw_vs_raw:
        print(f"RESOLVED topics are MORE stable (lower seed variation by {(1-mean_resolved_vs_resolved/mean_raw_vs_raw)*100:.1f}%)")
    else:
        print(f"RESOLVED topics are LESS stable (higher seed variation by {(mean_resolved_vs_resolved/mean_raw_vs_raw-1)*100:.1f}%)")

    # Create single comprehensive visualization
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram with overlays
        bins = np.linspace(min(min(kl_raw_vs_raw), min(kl_resolved_vs_resolved), min(kl_raw_vs_resolved)),
                          max(max(kl_raw_vs_raw), max(kl_resolved_vs_resolved), max(kl_raw_vs_resolved)), 15)
        
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
        labels = ['RAW vs RAW (Seed Variation)', 'RESOLVED vs RESOLVED (Seed Variation)', 'RAW vs RESOLVED (Pronoun Effect)']
        data_sets = [kl_raw_vs_raw, kl_resolved_vs_resolved, kl_raw_vs_resolved]
        means = [mean_raw_vs_raw, mean_resolved_vs_resolved, mean_raw_vs_resolved]
        
        # Plot histograms
        for data, color, label in zip(data_sets, colors, labels):
            ax.hist(data, bins=bins, alpha=0.6, color=color, label=label, edgecolor='black', linewidth=0.5)
        
        # Add mean lines
        for mean_val, color, label in zip(means, colors, labels):
            ax.axvline(mean_val, color=color, linestyle='--', linewidth=2.5, alpha=0.9)
            ax.text(mean_val, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.4f}', 
                   rotation=90, va='top', ha='right', fontsize=9, fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
        
        # Add effect size annotation
        y_pos = ax.get_ylim()[1] * 0.5
        ax.annotate('', xy=(mean_raw_vs_resolved, y_pos), xytext=(mean_resolved_vs_resolved, y_pos),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=2.5))
        ax.text((mean_raw_vs_resolved + mean_resolved_vs_resolved) / 2, y_pos + 0.3, 
               f'Effect: +{effect_above_resolved_noise:.4f} (+{effect_above_resolved_noise/mean_resolved_vs_resolved*100:.1f}%)',
               ha='center', va='bottom', fontsize=11, color='red', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='red', linewidth=2, alpha=0.9))
        
        ax.set_xlabel('KL Divergence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Topic Model Stability: KL Divergence Across Different Seeds', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('kl_across_runs.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\nSaved KL divergence comparison chart to kl_across_runs.png")
    except Exception as e:
        print(f"Failed to save KL runs chart: {e}")

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

            # Additional metric plots removed (cos_vals, skl_vals, js_vals, jaccard_per_topic not computed)
        except Exception as me:
            print(f"Failed to save metric density plots: {me}")
    except Exception as e:
        print(f"Failed to append KL summary: {e}")


if __name__ == "__main__":
    main()