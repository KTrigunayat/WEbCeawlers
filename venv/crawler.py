# crawler.py
import requests
import json
import os
import time
import base64 # For decoding file content
from config import GITHUB_TOKEN # Import your token

BASE_API_URL = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json" # Standard API version
}
DATA_OUTPUT_DIR = "github_data"

# --- Configuration for File Fetching ---
MAX_FILE_FETCH_DEPTH = 2  # How many directory levels deep to go for files
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1MB limit for individual files in prototype
ALLOWED_FILE_EXTENSIONS = ['.py', '.js', '.java', '.c', '.cpp', '.h', '.rb', '.go', '.rs', '.swift', '.kt', '.md', '.txt', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.sh']
# Set to None or empty list ALLOWED_FILE_EXTENSIONS = [] to allow all extensions (within size limits)

# --- Helper Function to Make API Requests ---
def make_api_request(url):
    """Makes a GET request to the GitHub API and handles basic rate limiting info."""
    print(f"Fetching: {url}")
    response = requests.get(url, headers=HEADERS)

    if 'X-RateLimit-Remaining' in response.headers:
        remaining = response.headers['X-RateLimit-Remaining']
        print(f"Rate limit remaining: {remaining}")
        if int(remaining) < 20: # Increased threshold for earlier warning
            print("WARNING: Low rate limit remaining! Pausing for a bit...")
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
            sleep_duration = max(0, reset_time - time.time()) + 5 # Sleep until reset + 5s buffer
            print(f"Sleeping for {sleep_duration:.2f} seconds until rate limit reset.")
            time.sleep(sleep_duration)


    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        print(f"Error 404: Resource not found at {url}")
        return None
    elif response.status_code == 403:
        print(f"Error 403: Forbidden. Check token or rate limits. Response: {response.text}")
        reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 300))
        sleep_duration = max(0, reset_time - time.time()) + 10 # Sleep until reset + 10s buffer
        print(f"Pausing for {sleep_duration:.2f} seconds due to 403 error...")
        time.sleep(sleep_duration)
        return None
    else:
        print(f"Error {response.status_code}: {response.text} for URL: {url}")
        return None

# --- Helper Function to Handle Pagination ---
def get_all_pages(url):
    """Fetches all items from a paginated GitHub API endpoint."""
    results = []
    next_url = url
    page_count = 1
    while next_url:
        print(f"Fetching paginated (page {page_count}): {next_url.split('?')[0]}...") # Keep URL cleaner
        response = requests.get(next_url, headers=HEADERS)

        if 'X-RateLimit-Remaining' in response.headers:
            remaining = response.headers['X-RateLimit-Remaining']
            # print(f"Rate limit remaining: {remaining}") # Less verbose for pagination
            if int(remaining) < 10:
                print("WARNING: Low rate limit during pagination! Pausing...")
                reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 60))
                sleep_duration = max(0, reset_time - time.time()) + 5
                print(f"Sleeping for {sleep_duration:.2f} seconds.")
                time.sleep(sleep_duration)

        if response.status_code == 200:
            page_data = response.json()
            if isinstance(page_data, list): # Ensure we got a list
                 results.extend(page_data)
            else:
                print(f"WARN: Expected list from paginated request, got {type(page_data)}. URL: {next_url}")
                break # Stop if response is not as expected

            if 'Link' in response.headers:
                links = requests.utils.parse_header_links(response.headers['Link'])
                next_url = None
                for link_info in links: # Renamed 'link' to 'link_info' to avoid conflict
                    if link_info['rel'] == 'next':
                        next_url = link_info['url']
                        break
            else:
                next_url = None
            page_count +=1
        elif response.status_code == 403:
            print(f"Error 403 on paginated request: {response.text}")
            reset_time = int(response.headers.get('X-RateLimit-Reset', time.time() + 300))
            sleep_duration = max(0, reset_time - time.time()) + 10
            print(f"Pausing for {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)
            break
        else:
            print(f"Error {response.status_code} on paginated request: {response.text}")
            break
    return results

# --- Function to Get User Profile ---
def get_user_profile(username):
    """Fetches the GitHub user's profile information."""
    url = f"{BASE_API_URL}/users/{username}"
    return make_api_request(url)

# --- Function to Get User Repositories ---
def get_user_repositories(username):
    """Fetches a user's public repositories."""
    url = f"{BASE_API_URL}/users/{username}/repos?type=owner&sort=updated&per_page=100"
    return get_all_pages(url)

# --- Function to Fetch File Content Recursively ---
def get_repo_file_contents_recursive(repo_full_name, path="", current_depth=0):
    """
    Recursively fetches file contents for a repository.
    Returns a dictionary where keys are file paths and values are their content.
    """
    if current_depth > MAX_FILE_FETCH_DEPTH:
        print(f"INFO: Max depth {MAX_FILE_FETCH_DEPTH} reached for path '{path}' in {repo_full_name}")
        return {}

    contents_url = f"{BASE_API_URL}/repos/{repo_full_name}/contents/{path}"
    # print(f"Fetching contents for: {contents_url} (Depth: {current_depth})") # Less verbose
    items = make_api_request(contents_url)
    time.sleep(0.25) # Reduced sleep, make_api_request has some sleep

    repo_files_data = {}

    if items is None:
        # print(f"WARN: No items found or error for {contents_url}") # make_api_request already prints
        return repo_files_data

    if isinstance(items, dict) and items.get("type") == "file": # If path was directly to a file
        items = [items]
    elif not isinstance(items, list):
        print(f"WARN: Expected a list of items for directory '{path}' in {repo_full_name}, got {type(items)}")
        return repo_files_data

    for item in items:
        item_type = item.get("type")
        item_path = item.get("path")
        item_name = item.get("name")

        if item_type == "file":
            file_extension = os.path.splitext(item_name)[1].lower() if item_name else ""
            if ALLOWED_FILE_EXTENSIONS and file_extension not in ALLOWED_FILE_EXTENSIONS:
                # print(f"INFO: Skipping file '{item_path}' due to extension: {file_extension}")
                continue

            if item.get("size", 0) > MAX_FILE_SIZE_BYTES:
                print(f"INFO: Skipping file '{item_path}' because it's too large: {item.get('size')} bytes")
                continue

            if 'content' in item and item.get('encoding') == 'base64':
                try:
                    file_content = base64.b64decode(item['content']).decode('utf-8', errors='replace')
                    repo_files_data[item_path] = file_content
                    # print(f"  Fetched content for: {item_path}")
                except Exception as e:
                    print(f"ERROR: Could not decode base64 content for {item_path}: {e}")
            elif item.get("download_url"):
                # print(f"  File '{item_path}' content not embedded, fetching via download_url...")
                raw_content_response = requests.get(item["download_url"], headers=HEADERS)
                time.sleep(0.1)
                if raw_content_response.status_code == 200:
                    try:
                        repo_files_data[item_path] = raw_content_response.text # Assumes text
                        # print(f"    Fetched raw content for: {item_path}")
                    except Exception as e:
                         print(f"ERROR: Could not decode raw content for {item_path}: {e}")
                else:
                    print(f"WARN: Could not download raw content for {item_path}. Status: {raw_content_response.status_code}")
            # else: # No need to print this warning, it's common for symlinks or submodules
                # print(f"WARN: No content or download_url for file {item_path}")

        elif item_type == "dir":
            if item_path:
                sub_dir_files = get_repo_file_contents_recursive(
                    repo_full_name,
                    path=item_path,
                    current_depth=current_depth + 1
                )
                repo_files_data.update(sub_dir_files)
        # time.sleep(0.05) # Reduced sleep further

    return repo_files_data

# --- Function to Get Repository Details ---
def get_repository_details(repo_full_name):
    """Fetches specific details for a repository, now including file contents."""
    repo_data = {}

    # 1. Get Languages
    lang_url = f"{BASE_API_URL}/repos/{repo_full_name}/languages"
    repo_data['languages'] = make_api_request(lang_url)
    time.sleep(0.5)

    # 2. Get Commits (Simplified: Get recent commits)
    commits_url = f"{BASE_API_URL}/repos/{repo_full_name}/commits?per_page=30"
    commits_response = make_api_request(commits_url)
    if commits_response and isinstance(commits_response, list):
        simplified_commits = []
        for commit_info in commits_response:
            if commit_info and isinstance(commit_info, dict) and \
               'commit' in commit_info and isinstance(commit_info['commit'], dict) and \
               'author' in commit_info['commit'] and isinstance(commit_info['commit']['author'], dict) and \
               'committer' in commit_info['commit'] and isinstance(commit_info['commit']['committer'], dict):
                simplified_commits.append({
                    'sha': commit_info.get('sha'),
                    'message': commit_info['commit'].get('message'),
                    'author_name': commit_info['commit']['author'].get('name'),
                    'author_date': commit_info['commit']['author'].get('date'),
                    'committer_name': commit_info['commit']['committer'].get('name'),
                    'committer_date': commit_info['commit']['committer'].get('date'),
                    'html_url': commit_info.get('html_url')
                })
            # else: # Be less verbose about malformed commits for prototype
                # print(f"Skipping potentially malformed commit in {repo_full_name}: {commit_info.get('sha', 'N/A') if isinstance(commit_info, dict) else 'N/A'}")
        repo_data['commits'] = simplified_commits
    else:
        repo_data['commits'] = []
    time.sleep(0.5)

    # 3. Get README
    readme_url = f"{BASE_API_URL}/repos/{repo_full_name}/readme"
    readme_data_response = make_api_request(readme_url)
    if readme_data_response and isinstance(readme_data_response, dict) and 'content' in readme_data_response:
        try:
            decoded_content = base64.b64decode(readme_data_response['content']).decode('utf-8', errors='replace')
            repo_data['readme_content'] = decoded_content
        except Exception as e:
            print(f"Error decoding README for {repo_full_name}: {e}")
            repo_data['readme_content'] = "Error decoding README or not UTF-8."
    else:
        repo_data['readme_content'] = None
    time.sleep(0.5)

    # 4. Get File Contents
    print(f"Fetching file contents for repository: {repo_full_name} (max depth {MAX_FILE_FETCH_DEPTH})...")
    repo_data['file_contents'] = get_repo_file_contents_recursive(repo_full_name)

    return repo_data

# --- Main Crawling Function ---
def crawl_github_user(username):
    """Main function to crawl a GitHub user and save data."""
    print(f"\n--- Starting crawl for user: {username} ---")
    user_data = {}

    profile = get_user_profile(username)
    if not profile:
        print(f"Could not fetch profile for {username}. Exiting.")
        return
    user_data['profile'] = profile
    time.sleep(0.5)

    print(f"\nFetching repositories for {username}...")
    repositories = get_user_repositories(username)
    if repositories is None:
        repositories = []

    user_data['repositories'] = []
    MAX_REPOS_TO_DETAIL = 5 # For prototype speed
    detailed_repos_count = 0

    if not repositories:
        print(f"No public repositories found for {username} or error fetching them.")

    for repo in repositories:
        if not isinstance(repo, dict): # Basic check
            print(f"Skipping invalid repo item: {repo}")
            continue

        repo_info = {
            'id': repo.get('id'),
            'name': repo.get('name'),
            'full_name': repo.get('full_name'),
            'html_url': repo.get('html_url'),
            'description': repo.get('description'),
            'fork': repo.get('fork'),
            'created_at': repo.get('created_at'),
            'updated_at': repo.get('updated_at'),
            'pushed_at': repo.get('pushed_at'),
            'stargazers_count': repo.get('stargazers_count'),
            'watchers_count': repo.get('watchers_count'),
            'forks_count': repo.get('forks_count'),
            'language': repo.get('language'),
            'open_issues_count': repo.get('open_issues_count'),
            'license': repo.get('license').get('name') if repo.get('license') and isinstance(repo.get('license'), dict) else None,
            'topics': repo.get('topics', [])
        }

        if not repo.get('fork') and detailed_repos_count < MAX_REPOS_TO_DETAIL:
            print(f"\nFetching details for repository: {repo.get('full_name')}...")
            if repo.get('full_name'): # Ensure full_name exists
                details = get_repository_details(repo['full_name'])
                repo_info.update(details)
            else:
                print(f"WARN: Repository item missing 'full_name': {repo.get('name')}")
            detailed_repos_count += 1
        elif repo.get('fork'):
            # print(f"Skipping details for forked repository: {repo.get('full_name')}") # Less verbose
            pass
        elif detailed_repos_count >= MAX_REPOS_TO_DETAIL:
            # print(f"Skipping details for {repo.get('full_name')} (reached max detailed repo limit)") # Less verbose
            pass

        user_data['repositories'].append(repo_info)
        time.sleep(0.1) # Small delay between processing each repo's basic info

    if not os.path.exists(DATA_OUTPUT_DIR):
        os.makedirs(DATA_OUTPUT_DIR)

    output_filename = os.path.join(DATA_OUTPUT_DIR, f"{username}_data.json")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=4, ensure_ascii=False)
        print(f"\nData for {username} saved to {output_filename}")
    except Exception as e:
        print(f"ERROR: Could not write JSON data to {output_filename}: {e}")

    print(f"--- Finished crawl for user: {username} ---")

# --- Script Entry Point ---
if __name__ == "__main__":
    # Create output directory if it doesn't exist at script start
    if not os.path.exists(DATA_OUTPUT_DIR):
        try:
            os.makedirs(DATA_OUTPUT_DIR)
        except OSError as e:
            print(f"ERROR: Could not create data directory {DATA_OUTPUT_DIR}: {e}")
            # Optionally exit if directory creation fails and is critical
            # import sys
            # sys.exit(1)

    target_username = input("Enter GitHub username to crawl: ")
    if target_username:
        crawl_github_user(target_username.strip()) # .strip() to remove accidental spaces
    else:
        print("No username provided.")