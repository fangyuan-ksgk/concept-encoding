from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import torch 
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm


def load_model():
    model_name = "HuggingFaceTB/SmolLM-360M-Instruct"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu"  # This will automatically handle GPU/CPU placement
    )
    
    return model, tokenizer

def get_concept_encoding(model, tokenizer, prompt: str, layer_idx: int, token_idx: int = -1):   
    inputs = tokenizer(prompt, return_tensors="pt")
    res = model(**inputs, output_hidden_states=True)
    return res.hidden_states[layer_idx][0, token_idx, :]

def get_concept_encodings(model, tokenizer, prompts: list[str], layer_idx: int, token_idx: int = -1):
    residual_streams = [get_concept_encoding(model, tokenizer, prompt, layer_idx, token_idx) for prompt in prompts]
    return torch.stack(residual_streams).detach().numpy()


def compute_concept_decodability(model, tokenizer, concept_examples: dict[str, list[str]], 
                               layer_idx: int, token_idx: int = -1, k: int = 10, n_shot: int = 5):
    """
    Compute per-concept decodability scores using k-NN classification
    Args:
        k: number of neighbors (default=10 as per paper)
    """
    X = []  # Features
    y = []  # Labels
    concept_indices = {}  # Track indices for each concept
    current_idx = 0
    
    for concept, examples in tqdm(concept_examples.items(), desc="Computing concept encodings"):
        encodings = get_concept_encodings(model, tokenizer, examples, layer_idx, token_idx)
        X.append(encodings)
        y.extend([concept] * len(examples))
        concept_indices[concept] = slice(current_idx, current_idx + len(examples))
        current_idx += len(examples)
    
    X = np.vstack(X)
    
    # Train k-NN classifier
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    
    # Calculate per-concept accuracy
    concept_scores = {}
    for concept in concept_examples:
        idx = concept_indices[concept]
        concept_scores[concept] = accuracy_score(y[idx], y_pred[idx])
    
    # pretty-print concept score information 
    print(f"Concept Decodability Scores for {n_shot}-shot ICL")
    for concept, score in concept_scores.items():
        print(f"{concept}: {score:.4f}")
    
    # Return scores and data for visualization
    return concept_scores, X, y

def visualize_concept_embeddings(X, y, title="Concept Embeddings", figsize=(10, 6)):
    """
    Visualize concept embeddings using t-SNE
    Args:
        X: numpy array of embeddings
        y: list/array of labels
        title: plot title
        figsize: figure size tuple
    """
    # Reduce dimensionality using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Get unique labels and assign different colors
    unique_labels = sorted(set(y))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    
    # Plot each concept cluster
    for label, color in zip(unique_labels, colors):
        mask = np.array(y) == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[color], 
                   label=label,
                   alpha=0.7,
                   s=50)  # s controls point size
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('concept_embeddings.png')  # Save to file
    plt.close()  # Clean up
    return plt

def plot_multiple_shots(model, tokenizer, concept_examples_list, layer_idx, 
                       shot_labels=[1, 2, 4, 6, 10], 
                       token_idx=-1, k=10):
    """
    Create multiple plots for different shot scenarios
    Args:
        concept_examples_list: list of concept_examples dictionaries for each shot scenario
        shot_labels: labels for each shot scenario
        Other args same as compute_concept_decodability
    """
    fig, axes = plt.subplots(1, len(concept_examples_list), 
                            figsize=(5*len(concept_examples_list), 4))
    
    for idx, (examples, shot_label) in enumerate(zip(concept_examples_list, shot_labels)):
        # Compute decodability and get embeddings
        scores, X, y = compute_concept_decodability(
            model, tokenizer, examples, layer_idx, token_idx, k, shot_label
        )
        
        # Create t-SNE plot on the corresponding subplot
        plt.sca(axes[idx])
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_2d = tsne.fit_transform(X)
        
        # Plot points
        unique_labels = sorted(set(y))
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = np.array(y) == label
            axes[idx].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                            c=[color], 
                            label=label,
                            alpha=0.7,
                            s=50)
        
        axes[idx].set_title(f"{shot_label}-shot ICL")
        if idx == len(concept_examples_list)-1:  # Only show legend for last plot
            axes[idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('multiple_shots.png')  # Save to file
    plt.close()  # Clean up
    return plt