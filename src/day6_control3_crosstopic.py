"""
MirrorProbe - Day 6 (continued) - Control 3 (Cross-Topic Generalization)
============================================================================
Groups the 50 topic_domains into broader clusters, then trains the
probe on all clusters EXCEPT one, testing only on the held-out cluster.
Repeated for every cluster. If accuracy stays well above chance on
held-out topics, the probe generalizes beyond memorized vocabulary.
"""

import json
import numpy as np
from sklearn.linear_model import LogisticRegression

with open("data/day3_activations.json", "r", encoding="utf-8-sig") as f:
    records = json.load(f)

# Map each specific topic_domain to a broader cluster
CLUSTER_MAP = {
    "school_subject_choice": "school", "exam_strategy": "school",
    "time_management": "school", "language_learning": "school",
    "math_logic": "school", "science_fact": "school",
    "history_fact": "school", "geography_fact": "school",
    "creative_writing_feedback": "school", "creative_writing_poem": "school",
    "code_logic_correctness": "school", "code_logic_correctness_2": "school",
    "school_uniform_cost": "school",

    "farming_decision": "rural_practical", "land_farming": "rural_practical",
    "drought_planning": "rural_practical", "livestock_care": "rural_practical",
    "tree_planting": "rural_practical", "water_labor_allocation": "rural_practical",
    "weather_prediction": "rural_practical",

    "friendship_money": "social", "social_conflict_decision": "social",
    "sibling_rivalry": "social", "friendship_secret_keeping": "social",
    "dance_competition": "social",

    "family_finance": "family", "religion_tradition": "family",
    "neighbor_dispute": "family",

    "health_hygiene": "health_safety", "herbal_medicine": "health_safety",
    "nutrition": "health_safety", "food_safety": "health_safety",
    "transportation": "health_safety", "bicycle_repair": "health_safety",
    "sports_training_plan": "health_safety", "sports_team_selection": "health_safety",

    "savings_stokvel": "economic", "debt_repayment": "economic",
    "business_trading_idea": "economic", "business_trading_idea_2": "economic",
    "market_price_negotiation": "economic", "job_career_choice": "economic",

    "technology_phone_use": "technical_modern", "app_game_choice": "technical_modern",
    "solar_electricity_plan": "technical_modern",

    "environment_conservation": "civic", "recycling_waste": "civic",
    "election_promise": "civic",

    "music_talent": "creative_other",
}

def build_dataset(layer_name, records, ids_filter=None):
    X, y = [], []
    for r in records:
        cluster = CLUSTER_MAP.get(r["topic_domain"], "other")
        if ids_filter is not None and cluster not in ids_filter:
            continue
        X.append(r["activations"][layer_name])
        y.append(1 if r["condition"] == "pressured" else 0)
    return np.array(X), np.array(y)

all_clusters = sorted(set(CLUSTER_MAP.values()))
print(f"Found {len(all_clusters)} topic clusters: {all_clusters}\n")

LAYER_COUNT = 12
# Focus on the layers that looked strongest/cleanest in Day 5/6 (6-10),
# but report all layers for completeness.
print("=" * 70)
print("CONTROL 3 - LEAVE-ONE-CLUSTER-OUT accuracy per layer")
print("(Each number = accuracy on a topic cluster NEVER seen in training)")
print("=" * 70)

results = {}
for layer in range(LAYER_COUNT):
    layer_name = f"layer_{layer}"
    layer_scores = []

    for held_out in all_clusters:
        train_clusters = [c for c in all_clusters if c != held_out]
        X_train, y_train = build_dataset(layer_name, records, ids_filter=train_clusters)
        X_test, y_test = build_dataset(layer_name, records, ids_filter=[held_out])

        if len(X_test) < 2 or len(set(y_test)) < 2:
            continue  # skip clusters too small to evaluate meaningfully

        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        layer_scores.append(acc)

    avg_acc = float(np.mean(layer_scores)) if layer_scores else None
    results[layer_name] = avg_acc
    print(f"Layer {layer:2d} | avg held-out-cluster accuracy: {avg_acc:.2f}" if avg_acc is not None else f"Layer {layer:2d} | not enough data")

with open("data/day6_control3_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to data/day6_control3_results.json")
print("\nINTERPRETATION: accuracy well above 0.50 on topics the probe")
print("NEVER trained on means it generalizes beyond memorized vocabulary.")
print("Accuracy near 0.50 means it likely just memorized topic-specific cues.")