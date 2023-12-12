import os
from huggingface_hub import HfApi, Repository
import toml

config_path = 'test_state.toml'


config = toml.load(config_path)

model_store = config["test"]["modelStore"]
model_directory = config["test"]["model_folder"]
repository_name = config["hf"]["hf_repo"]
hf_username = config["hf"]["hf_user"]
hf_access_token = config["hf"]["hf_tkn"]


def push_to_huggingface_hub(model_dir, model_repo_name, username, hf_token):
    """
    Pushes the model to Hugging Face Hub.

    Args:
    model_dir (str): Directory where the model is saved.
    model_repo_name (str): Name of the repository to be created on Hugging Face.
    username (str): Your Hugging Face username.
    hf_token (str): Your Hugging Face access token.
    """
    # Create a repository on Hugging Face Hub
    api = HfApi()
    api.create_repo(repo_id=model_repo_name, token=hf_token, private=False)

    # Clone, add, commit, and push files to the repository
    repo = Repository(local_dir=model_dir, clone_from=f"{username}/{model_repo_name}")
    repo.git_add()
    repo.git_commit("Initial commit with trained model")
    repo.git_push()

    print(f"Model successfully pushed to: https://huggingface.co/{username}/{model_repo_name}")




push_to_huggingface_hub(model_directory, repository_name, hf_username, hf_access_token)
