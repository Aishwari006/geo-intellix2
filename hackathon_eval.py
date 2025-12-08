import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# =======================================================
# SECTION A: IMPORT YOUR BACKEND HERE
# =======================================================
# 1. Look at your backend file name. Is it 'app.py'? 'model.py'? 
# 2. Look at the function name that predicts text. Is it 'predict()'?
# 3. Change the line below to match YOUR file and function.

# Example: from my_backend_script import generate_caption
# from your_backend_filename import your_prediction_function 

# --- PLACEHOLDER (Delete this function when you link your real one) ---
def mock_prediction_function(image_path):
    # This simulates what your model does. 
    # REPLACE THIS with your actual model call.
    return "A view of a flooded field with water covering crops."
# =======================================================

def calculate_detailed_bleu(reference_text, candidate_text):
    """
    Calculates BLEU 1, 2, 3, and 4 scores.
    """
    # 1. Tokenize (Split sentence into words)
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    cand_tokens = nltk.word_tokenize(candidate_text.lower())

    # BLEU expects a list of references [[ref]]
    references = [ref_tokens]

    # 2. Define Smoothing (Prevents score of 0 if a 4-gram is missing)
    smoother = SmoothingFunction().method1

    # 3. Calculate Scores with specific weights
    # BLEU-1: Checks individual words (Vocabulary)
    b1 = sentence_bleu(references, cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smoother)
    
    # BLEU-2: Checks pairs of words (Phrasing)
    b2 = sentence_bleu(references, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    
    # BLEU-3: Checks triplets (Fluency)
    b3 = sentence_bleu(references, cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    
    # BLEU-4: Checks 4-word sequences (Deep Structure)
    b4 = sentence_bleu(references, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return b1, b2, b3, b4

# =======================================================
# MAIN EXECUTION (Runs when you type 'python hackathon_eval.py')
# =======================================================
if __name__ == "__main__":
    
    # --- STEP 1: INPUTS FROM THE JUDGE ---
    # Update these two lines live during the demo!
    judge_image_path = "test_image.jpg"
    judge_ground_truth = "A flooded agricultural field with brown water covering the crops."

    print("\n" + "="*50)
    print("STARTING MODEL EVALUATION")
    print("="*50)
    print(f"[-] Image Path    : {judge_image_path}")
    print(f"[-] Ground Truth  : {judge_ground_truth}")

    # --- STEP 2: GENERATE PREDICTION ---
    print("[-] Running Model Inference...", end=" ", flush=True)
    
    # !!! UNCOMMENT THE LINE BELOW AND USE YOUR FUNCTION !!!
    # model_output = your_prediction_function(judge_image_path)
    
    # For now, using the mock function:
    model_output = mock_prediction_function(judge_image_path)
    
    print("DONE.")
    print(f"[-] Model Output  : {model_output}")
    print("-" * 50)

    # --- STEP 3: CALCULATE METRICS ---
    b1, b2, b3, b4 = calculate_detailed_bleu(judge_ground_truth, model_output)

    # --- STEP 4: DISPLAY REPORT ---
    print("\n--- 📊 FINAL METRICS REPORT ---")
    print(f"BLEU-1 (Vocabulary Match) : {b1*100:.2f}%")
    print(f"BLEU-2 (Phrase Match)     : {b2*100:.2f}%")
    print(f"BLEU-3 (Fluency Match)    : {b3*100:.2f}%")
    print(f"BLEU-4 (Structural Match) : {b4*100:.2f}%")
    print("="*50 + "\n")