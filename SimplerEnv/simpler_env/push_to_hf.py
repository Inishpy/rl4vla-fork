import argparse
from huggingface_hub import HfApi, create_repo, upload_folder

def push_model_to_hub(model_dir, repo_name, hf_token):
    api = HfApi()
    # Create repo if it doesn't exist
    try:
        create_repo(name=repo_name, token=hf_token, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation error (may already exist): {e}")

    # Upload the saved model folder
    try:
        upload_folder(
            repo_id=repo_name,
            folder_path=model_dir,
            token=hf_token,
            repo_type="model",
            commit_message=f"Upload model from {model_dir}"
        )
        print(f"Model pushed to Hugging Face Hub: {repo_name}")
    except Exception as e:
        print(f"Error pushing to Hugging Face Hub: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push a model directory to Hugging Face Hub.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the saved model directory")
    parser.add_argument("--repo_name", type=str, required=True, help="Hugging Face repo name (e.g. username/repo)")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    args = parser.parse_args()

    push_model_to_hub(args.model_dir, args.repo_name, args.hf_token)