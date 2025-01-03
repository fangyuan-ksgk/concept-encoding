icl_examples = {
    "Noun": [
        "dog",
        "computer",
        "book",
        "tree",
        "house",
        "car",
        "phone",
        "chair",
        "table",
        "bird",
        "cat",
        "school",
        "hospital",
        "mountain",
        "ocean",
        "coffee",
        "student",
        "teacher",
        "building",
        "garden"
    ],
    
    "Pronoun": [
        "he",
        "she",
        "it",
        "they",
        "we",
        "you",
        "I",
        "this",
        "that",
        "these",
        "those",
        "who",
        "whom",
        "whose",
        "which",
        "what",
        "myself",
        "yourself",
        "himself",
        "herself"
    ],
    
    "Verb": [
        "run",
        "jump",
        "eat",
        "sleep",
        "write",
        "read",
        "speak",
        "sing",
        "dance",
        "play",
        "work",
        "study",
        "teach",
        "learn",
        "walk",
        "swim",
        "fly",
        "drive",
        "cook",
        "build"
    ],
    
    "Adverb": [
        "quickly",
        "slowly",
        "carefully",
        "quietly",
        "loudly",
        "gently",
        "suddenly",
        "happily",
        "sadly",
        "easily",
        "hardly",
        "nearly",
        "almost",
        "really",
        "very",
        "too",
        "quite",
        "rather",
        "always",
        "never"
    ],
    
    "Adjective": [
        "tall",
        "short",
        "big",
        "small",
        "happy",
        "sad",
        "hot",
        "cold",
        "new",
        "old",
        "good",
        "bad",
        "fast",
        "slow",
        "hard",
        "soft",
        "bright",
        "dark",
        "clean",
        "dirty"
    ],
    
    "Preposition": [
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "to",
        "from",
        "under",
        "over",
        "between",
        "among",
        "through",
        "across",
        "behind",
        "beside",
        "during",
        "about",
        "above",
        "below"
    ]
}

import random   
from itertools import combinations


def build_icl_examples(n_shot: int = 5, cap_num: int = 100, icl_examples: dict = icl_examples):
    
    # Add NULL examples to ICL examples
    full_list = [] 
    for k in icl_examples: 
        full_list.extend(icl_examples[k])
    random.shuffle(full_list)
    icl_examples["NULL"] = full_list[:20]
    
    # Dictionary to store the results
    result = {}
    
    # For each category in icl_examples
    for k in icl_examples:
        # Get all possible combinations of n_shot examples
        all_combos = list(combinations(icl_examples[k], n_shot))
        # shuffle the combinations
        random.shuffle(all_combos)
        # select the first cap_num combinations and join words with commas
        result[k] = [", ".join(combo) for combo in all_combos[:cap_num]]
    
    return result

