import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

#Data Preperation:
def prepare_data(file_path:str) -> list:
    """
    Description:
    Function reads a file path to a json document with a JSON array of objects,
    each a question and answer pair for finetuning an LLM to adopt required behaviour foe a Kenyan 
    Legal Assistannt that provides basic legal information.

    Processes:
    1.Read in the data.
    2.Confirm count is greater than or equal to 100. (Sufficient training set)

    Args:
        path (str): The path to the data directory.
    """

    #KEY VARIABLES.
    #system_prompt - to be appended to every question_annswer pair
    #while building the LLaMA instruction-tuning JSONL format
    SYSTEM_PROMPT = (
        "You are a legal information assistant for a Kenyan legal-aid organisation, helping "
        "ordinary people understand their rights on land, tenancy, labour, and basic legal "
        "procedure under Kenyan law. For each question, identify the specific applicable Act "
        "and section, explain it in plain language, and apply it to the person's situation. "
        "Never invent rules, sections, or figures - if the facts given are insufficient, say "
        "so rather than guess, and note any important exceptions. Always end with a concrete "
        "next step, including when to consult an advocate, a specific court or tribunal, or "
        "the National Legal Aid Service (NLAS). You give general legal information only, "
        "never a binding legal judgment - make this clear in every response."
    )

    #KEY FUNCTIONS
    #Convert training example to LLaMA instruction-tuning JSONL format
    #Tokeniser - to count tokens 
    def _load_llama_tokenizer():
        """ Try to load the real LLaMA tokeniser. Requires HF access approval. 
        Returns None if unavailable so validation can fall back to an estimate."""
        try:
            from transformers import AutoTokenizer
            from dotenv import load_dotenv
            from huggingface_hub import login
            load_dotenv()
            token = os.getenv("HF_TOKEN") 
            if token:
                login(token=token)
            return AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        except Exception as exc:
            print(f"[WARN] Could not load the LLaMA tokeniser ({type(exc).__name__}).")
            print("[WARN] Falling back to an approximate token count (chars / 4).")
            print("[WARN] Re-run after your Hugging Face access is approved for exact counts.")
            return None

    def format_example(qa: dict) -> dict:
        return {
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": qa["question"]},
                {"role": "assistant", "content": qa["answer"]},
            ]
        }

    def count_tokens(msg: list, tokenizer) -> int:
        """Exact count via the LLaMA chat template, 
        or a documented approximation."""
        if tokenizer is not None:
            formatted = tokenizer.apply_chat_template(msg, tokenize=False)
            return len(tokenizer.encode(formatted))
        # Approximation: one token per ~4 characters of content plus template overhead
        chars = sum(len(m.get("content", "")) for m in msg.get("messages"))
        return (chars // 4) + 30

    def split_and_save(examples: list, output_dir: str = "data") -> None:
        """JSONL = one JSON object per line, the format Hugging Face datasets expects."""
        os.makedirs(output_dir, exist_ok=True)

        random.seed(42)  # Fixed seed: the same split every run, essential for debugging
        shuffled = examples[:]
        random.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)

        splits = [
            ("train", shuffled[:n_train]),
            ("val",   shuffled[n_train:n_train + n_val]),
            ("test",  shuffled[n_train + n_val:]),
        ]
        for split_name, split_data in splits:
            path = f"{output_dir}/{split_name}.jsonl"
            with open(path, "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex) + "\n")
            print(f"Saved {len(split_data)} examples to {path}")

        print(f"\nSplit summary: {len(splits[0][1])} train | {len(splits[1][1])} val | {len(splits[2][1])} test")



    #Read the data in and confirm if there are enough examples
    with open(file_path, "r") as f:
        data = json.load(f)
    if len(data) < 100:
        raise ValueError("Dataset too small: please provide at least 100 records.")

    print(len(data))

    #loop through each example and clean up
    #list to store the cleaned and transformed examples
    cleaned_transformed = []

    #Data prep report
    report = {
        "total": len(data), 
              "removed": [],
                "fixed": [], 
                "kept": 0
                }

    print("report object created")

    #Set to hold unique questions
    seen = set()

    for i, item in enumerate(data):
        # 1. Fix wrong key case BEFORE judging content
        if "Question" in item and "question" not in item:
            item["question"] = item.pop("Question")
            report["fixed"].append(f"Example {i}: renamed 'Question' to 'question'")


        # 2. Skip structurally malformed entries
        if "question" not in item or "answer" not in item:
            report["removed"].append(f"Example {i}: wrong format, skipped")
            continue


        # 3. Remove other keys that are not question or answer
        for key in list(item.keys()):
            if key not in ["question", "answer"]:
                item.pop(key)
                report["fixed"].append(f"Example {i}: removed extra key '{key}'")


        q = item.get("question", "").strip()
        a = item.get("answer", "").strip()

        # 4. Drop empty fields
        if not q or not a:
            report["removed"].append(f"Example {i}: empty field")
            continue


        # 5. Deduplicate on the question text
        if q in seen:
            report["removed"].append(f"Example {i}: duplicate question")
            continue
        seen.add(q)


        # 6. Minimum answer length: protocol answers must say something
        if len(a.split()) < 5:
            report["removed"].append(f"Example {i}: answer too short")
            continue


        # 7. Truncate runaway answers at 400 words
        words = a.split()
        if len(words) > 400:
            a = " ".join(words[:400])
            report["fixed"].append(f"Example {i}: truncated to 400 words")


        # Convert to LLaMA instruction-tuning JSONL format
        formatted_example = format_example(item)

        # # Check the token length
        # tokenizer = _load_llama_tokenizer()
        tokenizer = None
     

        tokens = count_tokens(formatted_example, tokenizer)
        if tokens < 64:
            report["removed"].append(f"Example {i}: very short ({tokens} tokens)")
        if tokens > 2048:
            report["removed"].append(f"Example {i}: too long ({tokens} tokens), exceeds training context")

    
        # Store clean examples in the "cleaned_transformed list"
        cleaned_transformed.append(formatted_example)

    #SPLITTING INTO TRAIN TEST AND VALIDATE SPLIT
    split_and_save(cleaned_transformed)  



    report["kept"] = len(cleaned_transformed)
    print(len(cleaned_transformed))
    print(cleaned_transformed[0])
    print(report.get("total"))
    print(report.get("removed"))
    print(report.get("fixed"))
    print(report.get("kept"))
   
    return cleaned_transformed, report

if __name__ == "__main__":
    file_path = f"{BASE_DIR}/raw_qa_pairs.json"
    prepare_data(file_path)
