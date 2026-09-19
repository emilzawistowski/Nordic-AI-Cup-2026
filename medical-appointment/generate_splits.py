import json
import random
from utils import group_questions_by_conversation

def main():
    conversations = group_questions_by_conversation()
    conv_ids = [conv_id for conv_id, _ in conversations]
    
    # Deterministic split using fixed seed
    random.seed(42)
    shuffled = conv_ids.copy()
    random.shuffle(shuffled)
    
    folds = [[] for _ in range(5)]
    for idx, conv_id in enumerate(shuffled):
        folds[idx % 5].append(conv_id)
        
    splits = {
        "seed": 42,
        "num_folds": 5,
        "folds": {
            f"fold_{i}": folds[i] for i in range(5)
        }
    }
    
    with open("experiments/splits.json", "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Generated 5 folds across {len(conv_ids)} conversations:")
    for fold_name, fold_convs in splits["folds"].items():
        print(f"  {fold_name}: {len(fold_convs)} conversations")

if __name__ == "__main__":
    main()
