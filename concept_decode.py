from utils import load_model, compute_concept_decodability, plot_multiple_shots
from data import build_icl_examples

model, tokenizer = load_model()

shots = [1, 2, 4, 6, 10]
concept_examples_list = [build_icl_examples(n_shot=shot, cap_num=100) for shot in shots]
concept_scores_list = [compute_concept_decodability(model, tokenizer, concept_examples, layer_idx=5, token_idx=-1) for concept_examples in concept_examples_list]
plot_multiple_shots(model, tokenizer, concept_examples_list, layer_idx=5, token_idx=-1)