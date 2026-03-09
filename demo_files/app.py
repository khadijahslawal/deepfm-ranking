import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


print("=== app.py starting ===")
import warnings
warnings.filterwarnings('ignore')

import re
import pickle
import numpy as np
import pandas as pd
import torch
import gradio as gr

from rank_bm25 import BM25Okapi
from sklearn.preprocessing import StandardScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# CHANGE FILENAMES HERE
# ── Load artifacts ────────────────────────────────────────────────────────────
print("Loading dataframe...")
df = pd.read_pickle('df_features_final.pkl')
print(f"Loaded dataframe: {df.shape}")

print("Loading scaler...")
with open('feature_scaler_final.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("Loading BM25 index...")
with open('bm25_index_final.pkl', 'rb') as f:
    bm25 = pickle.load(f)

# ── Build passage lookup ──────────────────────────────────────────────────────
# Create unique passage_id from query_id + passage_position
df['passage_id'] = df['query_id'].astype(str) + '_' + df['passage_position'].astype(str)
passage_lookup = (
    df[['passage_id', 'passage_text']]
    .drop_duplicates('passage_id')
    .set_index('passage_id')['passage_text']
    .to_dict()
)
pid_list = list(passage_lookup.keys())
print(f"Passage lookup: {len(passage_lookup):,} passages")

# ── Feature definitions ───────────────────────────────────────────────────────
DENSE_FEATURES = [
    'query_length', 'passage_length', 'length_ratio',
    'exact_match', 'query_term_coverage', 'jaccard_similarity',
    'tfidf_cosine_sim', 'bm25_score', 'passage_position',
    'jaccard_clean', 'term_coverage_clean', 'has_question_mark',
    'query_word_count', 'trigram_overlap', 'bm25_x_tfidf',
    'bm25_x_term_coverage', 'tfidf_x_jaccard', 'overlap_per_passage_len',
]
SPARSE_FEATURES = ['query_type']

# Filter to only columns present in the dataframe
DENSE_FEATURES  = [f for f in DENSE_FEATURES  if f in df.columns]
SPARSE_FEATURES = [f for f in SPARSE_FEATURES if f in df.columns]

# ── Rebuild DeepFM model ──────────────────────────────────────────────────────
print("Building DeepFM model...")
fixlen_feature_columns = (
    [SparseFeat(feat, vocabulary_size=df[feat].nunique() + 1, embedding_dim=4)
     for feat in SPARSE_FEATURES]
    +
    [DenseFeat(feat, 1) for feat in DENSE_FEATURES]
)

model = DeepFM(
    linear_feature_columns = fixlen_feature_columns,
    dnn_feature_columns    = fixlen_feature_columns,
    dnn_hidden_units       = (256, 128, 64),
    dnn_dropout            = 0.2,
    l2_reg_embedding       = 1e-4,
    l2_reg_linear          = 1e-4,
    l2_reg_dnn             = 1e-4,
    task                   = 'binary',
    device                 = DEVICE,
)
#CHANGE FILENAMES HERE
model.load_state_dict(torch.load('deepfm_msmarco_final.pth', map_location=DEVICE))
model.eval()

feature_names = get_feature_names(fixlen_feature_columns + fixlen_feature_columns)
print("DeepFM model loaded and ready.")

# ── Feature extraction helpers ────────────────────────────────────────────────
STOPWORDS = {
    'the','is','are','was','were','a','an','and','or','but','in','on',
    'at','to','for','of','with','by','from','that','this','it','as',
    'be','have','has','had','do','does','did','will','would','could',
    'should','may','might','what','which','who','how','when','where','why'
}

QUESTION_WORDS = {'what','when','where','who','why','how','which','is','are','can','does','do'}

def simple_tokenize(text: str) -> list:
    return re.sub(r'[^a-z0-9\s]', '', str(text).lower()).split()

def clean_tokens(text: str) -> set:
    return set(simple_tokenize(text)) - STOPWORDS

def get_query_type(query: str) -> int:
    first = str(query).lower().strip().split()[0] if query else ''
    if first == 'how':
        return 2
    if first in QUESTION_WORDS:
        return 1
    return 0

def extract_features_for_demo(query: str, passage: str,
                               bm25_score: float, passage_position: int) -> dict:
    q_tokens = simple_tokenize(query)
    p_tokens = simple_tokenize(passage)
    q_set    = set(q_tokens)
    p_set    = set(p_tokens)
    q_clean  = clean_tokens(query)
    p_clean  = clean_tokens(passage)

    q_len = max(len(q_tokens), 1)
    p_len = max(len(p_tokens), 1)

    matched        = q_set & p_set
    matched_clean  = q_clean & p_clean
    term_coverage  = len(matched) / q_len
    term_cov_clean = len(matched_clean) / max(len(q_clean), 1)
    jaccard        = len(matched) / max(len(q_set | p_set), 1)
    jaccard_cln    = len(matched_clean) / max(len(q_clean | p_clean), 1)
    exact          = int(query.lower() in passage.lower())

    if len(q_tokens) >= 3:
        ngrams  = [' '.join(q_tokens[i:i+3]) for i in range(len(q_tokens)-2)]
        trigram = sum(1 for ng in ngrams if ng in passage.lower()) / len(ngrams)
    else:
        trigram = term_cov_clean

    tfidf_sim = 0.0  # placeholder — model degrades gracefully

    feats = {
        'query_length'           : q_len,
        'passage_length'         : p_len,
        'length_ratio'           : q_len / p_len,
        'exact_match'            : exact,
        'query_term_coverage'    : term_coverage,
        'jaccard_similarity'     : jaccard,
        'tfidf_cosine_sim'       : tfidf_sim,
        'bm25_score'             : bm25_score,
        'passage_position'       : passage_position,
        'jaccard_clean'          : jaccard_cln,
        'term_coverage_clean'    : term_cov_clean,
        'has_question_mark'      : int('?' in query),
        'query_word_count'       : q_len,
        'trigram_overlap'        : trigram,
        'bm25_x_tfidf'           : bm25_score * tfidf_sim,
        'bm25_x_term_coverage'   : bm25_score * term_cov_clean,
        'tfidf_x_jaccard'        : tfidf_sim * jaccard_cln,
        'overlap_per_passage_len': term_cov_clean / (p_len + 1),
        'query_type'             : get_query_type(query),
    }
    return {k: feats[k] for k in feats if k in DENSE_FEATURES + SPARSE_FEATURES}

# ── Inference pipeline ────────────────────────────────────────────────────────
def rank_passages(query: str, top_k: int = 10) -> pd.DataFrame:
    query_tokens = simple_tokenize(query)
    scores       = bm25.get_scores(query_tokens)
    top_indices  = np.argsort(scores)[::-1][:top_k]

    rows = []
    for rank, idx in enumerate(top_indices):
        pid        = pid_list[idx]
        passage    = passage_lookup.get(pid, '')
        bm25_score = float(scores[idx])
        feats      = extract_features_for_demo(query, passage, bm25_score, rank)
        feats['passage_text'] = passage
        rows.append(feats)

    results_df = pd.DataFrame(rows)

    dense_cols = [c for c in DENSE_FEATURES if c in results_df.columns]
    results_df[dense_cols] = scaler.transform(
        results_df[dense_cols].astype(np.float32)
    )

    model_input = {}
    for name in feature_names:
        if name in results_df.columns:
            model_input[name] = results_df[name].values
        else:
            model_input[name] = np.zeros(len(results_df), dtype=np.float32)

    deepfm_scores              = model.predict(model_input, batch_size=256)
    results_df['deepfm_score'] = deepfm_scores
    results_df                 = results_df.sort_values('deepfm_score', ascending=False)
    results_df['deepfm_rank']  = range(1, len(results_df) + 1)

    return results_df[['deepfm_rank', 'deepfm_score', 'passage_text']]

# ── Gradio handler ────────────────────────────────────────────────────────────
def gradio_search(query: str, top_k: int) -> tuple:
    if not query or not query.strip():
        empty = pd.DataFrame(columns=['Rank', 'DeepFM Score', 'Passage'])
        return empty, "Please enter a query."

    try:
        results = rank_passages(query.strip(), top_k=int(top_k))

        deepfm_out = results[['deepfm_rank', 'deepfm_score', 'passage_text']].copy()
        deepfm_out.columns = ['Rank', 'DeepFM Score', 'Passage']
        deepfm_out['DeepFM Score'] = deepfm_out['DeepFM Score'].round(4)
        deepfm_out = deepfm_out.reset_index(drop=True)

        top_passage = results.iloc[0]['passage_text'][:200]
        summary = (
            f"Query: '{query}'\n"
            f"Retrieved {len(results)} candidates via BM25, re-ranked by DeepFM.\n"
            f"Top passage (truncated): {top_passage}..."
        )

        return deepfm_out, summary

    except Exception as e:
        empty = pd.DataFrame(columns=['Rank', 'DeepFM Score', 'Passage'])
        return empty, f"Error: {str(e)}"

# ── Example queries ───────────────────────────────────────────────────────────
EXAMPLE_QUERIES = [
    ["what is the capital of france", 5],
    ["how does photosynthesis work", 5],
    ["symptoms of diabetes", 10],
    ["how to train a neural network", 10],
    ["what causes inflation", 5],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="DeepFM Search Re-Ranker", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🔍 DeepFM Search Re-Ranker
    ### MS MARCO Passage Ranking Demo
    Enter a query to retrieve and re-rank candidate passages using **DeepFM**.
    """)

    with gr.Row():
        query_input = gr.Textbox(
            label       = "Search Query",
            placeholder = "e.g. what causes inflation",
            scale       = 4
        )
        top_k_slider = gr.Slider(
            minimum = 3,
            maximum = 20,
            value   = 10,
            step    = 1,
            label   = "Top-K candidates",
            scale   = 1
        )

    search_btn  = gr.Button("Search & Re-Rank", variant="primary")
    summary_box = gr.Textbox(label="Summary", interactive=False, lines=3)

    gr.Markdown("### 🤖 DeepFM Re-Ranked Results")
    deepfm_table = gr.Dataframe(
        headers  = ['Rank', 'DeepFM Score', 'Passage'],
        datatype = ['number', 'number', 'str'],
        wrap     = True,
    )

    gr.Examples(
        examples = EXAMPLE_QUERIES,
        inputs   = [query_input, top_k_slider],
        label    = "Example Queries"
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. BM25 retrieves the top-K candidate passages from the MS MARCO corpus
    2. DeepFM scores each candidate using learned feature interactions
    3. Results are returned sorted by DeepFM relevance score
    """)

    search_btn.click(
        fn      = gradio_search,
        inputs  = [query_input, top_k_slider],
        outputs = [deepfm_table, summary_box]
    )

    query_input.submit(
        fn      = gradio_search,
        inputs  = [query_input, top_k_slider],
        outputs = [deepfm_table, summary_box]
    )

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch()