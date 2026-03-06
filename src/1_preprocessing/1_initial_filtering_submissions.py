import subprocess

# Define the command as a list of arguments
command = [
    "python3", "combine_folder_multiprocess.py",
    "../../../../data/reddit/submissions",
    "--field", "subreddit",
    "--value", "AskDocs",
    "--file_filter", "202(5|6)",
    "--output", "../../output/corpora/submissions"
]

# Run the command
subprocess.run(command, check=True)
