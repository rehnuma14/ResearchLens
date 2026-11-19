import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.engine import load_and_vectorize, evaluate_idea

print("=" * 75)
print("UPRESEARCHLENS CLI VERIFICATION & EXPORT")
print("=" * 75)

data_path = os.path.join(BASE_DIR, "data", "papers.csv")
df, vectorizer, tfidf_matrix = load_and_vectorize(data_path)

print(f"1. Indexed Dataset     : {len(df)} curated real publications")
print(f"2. TF-IDF Matrix Shape : {tfidf_matrix.shape} (Documents x Features)")
print(f"3. Active Vocabulary   : {len(vectorizer.get_feature_names_out())} unique n-grams\n")

test_idea = "Explainable AI for Bangla fake news detection using LIME and SHAP"
result = evaluate_idea(test_idea, df, vectorizer, tfidf_matrix, top_k=3)

print(f"Tested Hypothesis: \"{test_idea}\"\n")
print(f"Novelty Score     : {result['novelty_score']}% ({result['novelty_label']})")
print(f"Feasibility Score : {result['feasibility_score']}%\n")
print("Top Literature Matches:")
for _, r in result["top_matches"].iterrows():
    print(f" • [Sim: {r['similarity']:.4f}] #{r['id']} ({r['year']}) {r['title']}")
    print(f"   Link: {r['source_link']}")

out_dir = os.path.join(BASE_DIR, "outputs")
os.makedirs(out_dir, exist_ok=True)
csv_out = os.path.join(out_dir, "latest_matches.csv")
result["top_matches"].to_csv(csv_out, index=False)
print(f"\n[OK] Results written to: {csv_out}")
print("=" * 75)