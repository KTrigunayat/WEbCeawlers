import re
import requests
import json
import os
from typing import List, Dict, Any, Optional, Tuple

# Note: For full functionality, you would need to install these libraries.
# pip install PyPDF2 python-docx mistralai-client PyGithub selenium beautifulsoup4 webdriver-manager
from PyPDF2 import PdfReader
from docx import Document
from mistralai.client import MistralClient
from mistralai.models import ChatCompletionChoice  # ✅ This is the correct way

from github import Github # Note: This specific Github object from PyGithub is not used in the current active API logic,
                          # which uses direct requests. It's uncommented as per request.
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service # For webdriver-manager
# from webdriver_manager.chrome import ChromeDriverManager # For automatically managing ChromeDriver
from bs4 import BeautifulSoup

# --- I. Resume Processing ---#
def download_resume(source_url: str, download_folder: str = "resumes") -> Optional[str]:
    """Downloads a resume from a URL to a local folder.
    Returns the file path if successful, otherwise None."""
    if not os.path.exists(download_folder):
        try:
            os.makedirs(download_folder)
        except OSError as e:
            print(f"Error creating download folder {download_folder}: {e}")
            return None
    try:
        response = requests.get(source_url, stream=True, timeout=10)
        response.raise_for_status()
        
        filename = source_url.split('/')[-1]
        if 'content-disposition' in response.headers:
            disp = response.headers['content-disposition']
            fn_match = re.search(r'filename="?([^"]+)"?', disp)
            if fn_match:
                filename = fn_match.group(1)
        
        filename = re.sub(r'[^\w\.\-]', '_', filename) # Sanitize
        if not filename: filename = "downloaded_resume"
        
        if '.' not in filename:
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' in content_type: filename += ".pdf"
            elif 'word' in content_type and ('docx' in content_type or 'officedocument' in content_type): filename += ".docx"
            else: filename += ".dat" # Default extension if not determinable

        file_path = os.path.join(download_folder, filename)
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Resume downloaded successfully to: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading resume from {source_url}: {e}")
        return None
    except IOError as e:
        print(f"Error saving resume: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download_resume: {e}")
        return None

def _parse_pdf_resume(file_path: str) -> str:
    """Helper to parse PDF resume text. Requires PyPDF2."""
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF {file_path} (PyPDF2): {e}")
        return "" # Return empty string on error
    return text

def _parse_docx_resume(file_path: str) -> str:
    """Helper to parse DOCX resume text. Requires python-docx."""
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {file_path} (python-docx): {e}")
        return "" # Return empty string on error
    return text

def extract_text_from_resume(file_path: str) -> str:
    """Extracts all text from a resume file (supports PDF, DOCX, TXT).
    Returns the extracted text as a string."""
    text = ""
    _, extension = os.path.splitext(file_path.lower())
    try:
        if extension == ".pdf":
            text = _parse_pdf_resume(file_path)
        elif extension == ".docx":
            text = _parse_docx_resume(file_path)
        elif extension == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        else:
            print(f"Unsupported file type: {extension} for {file_path}. Attempting plain text read.")
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e_txt:
                 print(f"Could not read {file_path} as plain text: {e_txt}")
                 return ""
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return ""
    return text

def extract_contact_info_from_text(resume_text: str) -> Dict[str, Optional[str]]:
    """Extracts contact information (email, phone) from resume text using regex.
    Returns a dictionary with 'email' and 'phone'."""
    contacts = {"email": None, "phone": None}
    if not resume_text: return contacts

    # Improved email regex (more common)
    email_regex = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_match = re.search(email_regex, resume_text)
    if email_match:
        contacts["email"] = email_match.group(0)
    
    # Improved phone regex to capture more formats
    phone_regex = r"\(?\b\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b" # Basic US
    # Consider more international formats if needed
    # phone_regex = r"(?:(?:\+?1\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?"
    phone_match = re.search(phone_regex, resume_text)
    if phone_match:
        contacts["phone"] = phone_match.group(0)
    return contacts

def extract_links_from_text(resume_text: str) -> List[str]:
    """Extracts all URLs from resume text using regex.
    Returns a list of unique found URLs."""
    if not resume_text: return []
    # More robust URL regex
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    links = re.findall(url_regex, resume_text)
    return list(set(links)) # Return unique links

def identify_github_link(links: List[str]) -> Optional[str]:
    """Identifies a GitHub profile link from a list of URLs.
    Returns the GitHub URL string or None if not found."""
    if not links: return None
    github_links = [link for link in links if "github.com/" in link.lower()]
    for link in github_links:
        try:
            parsed_url = requests.utils.urlparse(link)
            path_parts = [part for part in parsed_url.path.split('/') if part]
            # Check if it's a user profile (e.g., github.com/username)
            if len(path_parts) == 1 and not any(ext in path_parts[0] for ext in ['.git', '.zip', 'issues', 'pulls', 'actions', 'projects', 'wiki', 'security', 'insights', 'settings', 'orgs', 'sponsors', 'stars', 'followers', 'following', 'repositories', 'packages', 'gists', 'topics']):
                return link.strip().rstrip('/') # Likely a username
        except Exception:
            continue # Ignore malformed URLs
    if github_links: # Fallback to first found GitHub link if no clear profile link
        return github_links[0].strip().rstrip('/')
    return None

# --- II. LLM Interaction ---

def summarize_with_mistral(text_to_summarize: str, mistral_api_key: str) -> str:
    """Summarizes text using the Mistral LLM API."""
    if not text_to_summarize: return "No text provided for summarization."
    if not mistral_api_key: return "Mistral API key not provided. Cannot summarize."
    try:
        client = MistralClient(api_key=mistral_api_key)
        messages = [ChatMessage(role="user", content=f"Summarize this resume: {text_to_summarize}")]
        # Make sure to use a model that is available and suitable for summarization
        chat_response = client.chat(model="mistral-small-latest", messages=messages) # or other suitable model
        return chat_response.choices[0].message.content
    except Exception as e:
        print(f"Error during Mistral summarization: {e}")
        return f"Error summarizing text. (Original first 100 chars: {text_to_summarize[:100]}...)"

def compare_text_with_jd_llm(text_content: str, job_description: str, mistral_api_key: str) -> float:
    """Compares text with JD using an LLM for relevance.
    Returns a relevance score (0.0 to 1.0)."""
    if not text_content or not job_description: return 0.0
    if not mistral_api_key:
        print("Mistral API key not provided. Cannot compare with JD.")
        return 0.5 # Default if API key missing

    try:
        client = MistralClient(api_key=mistral_api_key)
        prompt = (
            f"Job Description:\n{job_description}\n\n"
            f"Candidate Text:\n{text_content}\n\n"
            "Based on the Job Description and the Candidate Text, how relevant is the candidate? "
            "Respond ONLY with a numerical score from 0.0 (not relevant at all) to 1.0 (perfectly relevant)."
        )
        messages = [ChatMessage(role="user", content=prompt)]
        chat_response = client.chat(model="mistral-small-latest", messages=messages) # or other suitable model
        
        response_text = chat_response.choices[0].message.content.strip()
        # Try to extract a float from the response
        score_match = re.search(r"(\d\.\d+)", response_text)
        if score_match:
            return float(score_match.group(1))
        else: # Fallback if LLM doesn't give just a number
            print(f"Could not parse float score from LLM response: '{response_text}'. Using default.")
            return 0.5  # Default on parse error or non-numeric response
    except ValueError as ve:
        print(f"Error parsing LLM score: {response_text} - {ve}")
        return 0.5 # Default on parse error
    except Exception as e:
        print(f"Error during Mistral comparison: {e}")
        return 0.0 # Default on API or other error

# --- III. Resume Scoring ---
def score_resume_ats(resume_summary: str, job_description: str, mistral_api_key: str) -> int:
    """Scores the resume summary against the job description using an LLM.
    Returns a score out of 100."""
    relevance_score = compare_text_with_jd_llm(resume_summary, job_description, mistral_api_key)
    return int(relevance_score * 100)

# --- IV. GitHub Data Fetching & Validation ---
def validate_github_url(github_url: str) -> Optional[str]:
    """Validates GitHub URL format and determines if it's a user profile or repository.
    Returns 'user', 'repo', or None if invalid or not identifiable."""
    if not github_url or "github.com" not in github_url.lower():
        return None
    try:
        parsed_url = requests.utils.urlparse(github_url)
        if parsed_url.netloc.lower() not in ["github.com", "www.github.com"]:
            return None
        path_parts = [part for part in parsed_url.path.split('/') if part]
        
        # Common non-user/repo paths at the first level
        common_non_user_paths = [
            "topics", "trending", "explore", "marketplace", "sponsors", "settings", 
            "notifications", "new", "organizations", "codespaces", "issues", "pulls", 
            "login", "join", "about", "pricing", "contact", "features", "customer-stories",
            "enterprise", "team", "security", "blog", "careers", "press", "search"
        ]
        
        if not path_parts: return None # e.g., just "github.com"

        if len(path_parts) == 1 and path_parts[0].lower() not in common_non_user_paths and '.' not in path_parts[0]:
            return "user"
        # Common non-repo paths at the second level when first is a user
        user_specific_non_repo_paths = [
            "stars", "followers", "following", "repositories", "projects", 
            "packages", "gists", "sponsoring", "tab" # common for user profile tabs
        ]
        if len(path_parts) == 2 and \
           path_parts[0].lower() not in common_non_user_paths and \
           '.' not in path_parts[0] and \
           path_parts[1].lower() not in user_specific_non_repo_paths and \
           '.' not in path_parts[1]: # Basic check to avoid file names in paths
            return "repo"
            
    except Exception as e:
        print(f"Error validating GitHub URL {github_url}: {e}")
        return None
    return None # Default if not clearly user or repo

def fetch_github_api_data(api_url: str, github_token: Optional[str]) -> Optional[Any]:
    """Helper function to fetch data from GitHub API.
    Returns JSON response as dict/list, or None on error."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"GitHub API HTTP error for {api_url}: {e.response.status_code} - {e.response.text[:200]}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"GitHub API request failed for {api_url}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON from GitHub API {api_url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching GitHub API data for {api_url}: {e}")
        return None


def get_github_username_from_url(github_url: str) -> Optional[str]:
    """Extracts GitHub username from a GitHub profile URL.
    Returns username string or None."""
    if not github_url: return None
    try:
        parsed_url = requests.utils.urlparse(github_url)
        if parsed_url.netloc.lower() in ["github.com", "www.github.com"]:
            path_parts = [part for part in parsed_url.path.split('/') if part]
            # Common non-user paths that might appear as the first part
            common_non_user_paths = [
                "topics", "trending", "explore", "marketplace", "sponsors", "settings", 
                "notifications", "new", "organizations", "codespaces", "issues", "pulls",
                 "login", "join", "about", "pricing", "contact", "features", "customer-stories",
                "enterprise", "team", "security", "blog", "careers", "press", "search"
            ]
            if len(path_parts) >= 1 and path_parts[0].lower() not in common_non_user_paths and '.' not in path_parts[0]:
                return path_parts[0]
    except Exception as e:
        print(f"Error parsing GitHub username from URL {github_url}: {e}")
        return None
    return None

def fetch_github_user_data_api(username: str, github_token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Fetches user profile data from GitHub API.
    Returns user data dictionary or None."""
    if not username: return None
    api_url = f"https://api.github.com/users/{username}"
    return fetch_github_api_data(api_url, github_token)

def fetch_github_user_repos_api(username: str, github_token: Optional[str], per_page: int = 30, page: int = 1) -> Optional[List[Dict[str, Any]]]:
    """Fetches a user's public repositories from GitHub API.
    Returns list of repo dictionaries or None."""
    if not username: return None
    api_url = f"https://api.github.com/users/{username}/repos?per_page={per_page}&page={page}&sort=updated&type=owner" # Added type=owner to get non-forks primarily
    return fetch_github_api_data(api_url, github_token)

def fetch_github_repo_languages_api(languages_url: str, github_token: Optional[str]) -> Optional[Dict[str, int]]:
    """Fetches language breakdown for a specific repository from its languages_url.
    Returns dictionary of languages and their byte counts or None."""
    if not languages_url: return None
    return fetch_github_api_data(languages_url, github_token)

def fetch_github_data_with_browser_agent(github_url: str) -> Optional[str]:
    """Fetches GitHub page content using a web browser agent (Selenium).
    Actual Selenium/BeautifulSoup implementation needed for complex scraping."""
    if not github_url: return None
    
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu') # Optional, sometimes useful
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")


    driver = None
    try:
        # Option 1: ChromeDriver in PATH
        driver = webdriver.Chrome(options=options)
        # Option 2: Using webdriver_manager (install with: pip install webdriver-manager)
        # driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        # Option 3: Explicit path to ChromeDriver
        # driver = webdriver.Chrome(executable_path='/path/to/your/chromedriver', options=options)
        
        driver.get(github_url)
        # Add explicit waits if dynamic content loading is an issue
        # from selenium.webdriver.support.ui import WebDriverWait
        # from selenium.webdriver.support import expected_conditions as EC
        # from selenium.webdriver.common.by import By
        # WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        page_source = driver.page_source
        return page_source
    except Exception as e:
        print(f"Error fetching {github_url} with Selenium: {e}")
        # You might want to check if the error is due to ChromeDriver not being found/configured
        if "executable needs to be in PATH" in str(e).lower():
            print("Hint: ChromeDriver might not be in your PATH or correctly configured.")
        return None
    finally:
        if driver:
            driver.quit()

# --- V. GitHub Analysis & Scoring ---

def analyze_repo_languages_from_data(repo_languages_data: Dict[str, int]) -> Dict[str, int]:
    """Analyzes and returns the language breakdown from fetched GitHub repo language data.
    Returns a dictionary of languages and their byte counts."""
    return repo_languages_data if repo_languages_data else {}

def check_relevance_of_languages(used_languages: List[str], jd_expected_languages: List[str]) -> float:
    """Checks relevance of coding languages used against JD requirements.
    Returns a relevance score (0.0 to 1.0). Case-insensitive matching."""
    if not jd_expected_languages: return 1.0 # No expected languages, so all are relevant (or vacuously true)
    if not used_languages: return 0.0
    
    used_set = set(lang.lower().strip() for lang in used_languages)
    expected_set = set(lang.lower().strip() for lang in jd_expected_languages)
    
    common_languages = used_set.intersection(expected_set)
    
    # Score based on how many of the *expected* languages are found
    return len(common_languages) / len(expected_set) if expected_set else 0.0

def analyze_repo_description_relevance(repo_description: Optional[str], job_description: str, mistral_api_key: str) -> float:
    """Analyzes a repository's description for relevance to the Job Description using an LLM.
    Returns a relevance score (0.0 to 1.0)."""
    if not repo_description:
        return 0.0
    # Using the generic comparison function
    return compare_text_with_jd_llm(repo_description, job_description, mistral_api_key)

def check_if_fork(repo_data: Dict[str, Any]) -> bool:
    """Checks if a repository is a fork based on GitHub API data.
    Returns True if it's a fork, False otherwise."""
    return repo_data.get("fork", False) if repo_data else False

def rate_code_quality_placeholder(repo_full_name: str, github_token: Optional[str]) -> float:
    """Placeholder for a complex code quality rating function.
    This would involve cloning repo, static analysis (e.g. SonarQube, pylint, etc.),
    checking for tests, documentation, commit hygiene, etc.
    Returns score (0.0 to 1.0)."""
    print(f"Placeholder: Assessing code quality for {repo_full_name} (token: {'*' * len(github_token) if github_token else 'None'}). Complex implementation needed (e.g., static analysis, test coverage).")
    # For a real implementation, you might:
    # 1. Clone the repo (if public or with token access)
    # 2. Run static analysis tools.
    # 3. Check for a test suite and estimate coverage.
    # 4. Look for CI/CD setup.
    # This is a very simplified placeholder.
    return 0.65 # Example placeholder score

def score_github_profile_aspects(username: str, github_token: Optional[str], job_description: str, mistral_api_key: str, jd_expected_languages: List[str]) -> Dict[str, float]:
    """Scores various aspects of a GitHub profile.
    Returns a dictionary of aspect scores (each 0-100)."""
    scores = {
        "profile_completeness": 0.0, "project_relevance": 0.0, "language_match": 0.0,
        "code_quality_avg": 0.0, "activity_level": 0.0, "original_projects_ratio": 0.0
    }
    if not username: return {key: 0.0 for key in scores}

    user_data = fetch_github_user_data_api(username, github_token)
    # Fetch more repos to get a better sample, especially non-forks
    user_repos_data = fetch_github_user_repos_api(username, github_token, per_page=30) 

    if not user_data: return {key: 0.0 for key in scores}

    # Profile Completeness
    completeness_score = 0
    if user_data.get("name"): completeness_score += 30
    if user_data.get("bio"): completeness_score += 30
    if user_data.get("company"): completeness_score += 15
    if user_data.get("blog"): completeness_score += 15 # Personal website/blog
    if user_data.get("location"): completeness_score += 10
    scores["profile_completeness"] = min(completeness_score, 100.0)

    # Activity Level (simple heuristic)
    # Consider recent activity, not just counts. GitHub API provides `pushed_at` for repos.
    # For now, using public_repos and followers as a proxy.
    public_repos_count = user_data.get("public_repos", 0)
    followers_count = user_data.get("followers", 0)
    # Normalize: e.g., 1 point per repo up to 50, 0.2 per follower up to 250
    activity_score = min(public_repos_count, 50) + min(followers_count * 0.2, 50)
    scores["activity_level"] = min(activity_score, 100.0)


    if user_repos_data:
        all_user_repos = user_repos_data
        non_fork_repos = [repo for repo in all_user_repos if repo and not check_if_fork(repo)]
        
        scores["original_projects_ratio"] = (len(non_fork_repos) / len(all_user_repos)) * 100 if all_user_repos else 0.0

        # Analyze a limited number of recent, non-forked repositories for depth
        repos_to_analyze = sorted(
            non_fork_repos, 
            key=lambda r: r.get("pushed_at", "1970-01-01T00:00:00Z"), 
            reverse=True
        )[:5] # Analyze top 5 most recently pushed non-forked repos

        relevant_project_scores = []
        code_quality_scores = []
        all_repo_languages_bytes: Dict[str, int] = {}

        for repo in repos_to_analyze:
            if not repo: continue
            
            desc = repo.get("description")
            if desc: # Only consider repos with descriptions for relevance scoring
                desc_relevance = analyze_repo_description_relevance(desc, job_description, mistral_api_key)
                relevant_project_scores.append(desc_relevance)
            
            # Placeholder for code quality
            if repo.get("full_name"):
                quality = rate_code_quality_placeholder(repo["full_name"], github_token)
                code_quality_scores.append(quality)

            if repo.get("languages_url"):
                repo_langs = fetch_github_repo_languages_api(repo["languages_url"], github_token)
                if repo_langs:
                    for lang, size in repo_langs.items():
                        all_repo_languages_bytes[lang] = all_repo_languages_bytes.get(lang, 0) + size
        
        if relevant_project_scores: 
            scores["project_relevance"] = (sum(relevant_project_scores) / len(relevant_project_scores)) * 100
        if code_quality_scores: 
            scores["code_quality_avg"] = (sum(code_quality_scores) / len(code_quality_scores)) * 100
        
        if all_repo_languages_bytes:
            # Sort languages by byte size (usage)
            sorted_languages = sorted(all_repo_languages_bytes.items(), key=lambda item: item[1], reverse=True)
            # Consider top N languages or all if fewer than N
            top_languages = [lang[0] for lang in sorted_languages[:5]] 
            scores["language_match"] = check_relevance_of_languages(top_languages, jd_expected_languages) * 100
    else:
        print(f"No repository data found for user {username} to analyze.")

    return scores

def calculate_overall_github_score(github_aspect_scores: Dict[str, float]) -> int:
    """Calculates an overall GitHub score by weighted average of aspect scores.
    Returns a score out of 100."""
    if not github_aspect_scores: return 0
    
    weights = {
        "project_relevance": 0.30,
        "language_match": 0.25,
        "code_quality_avg": 0.20, # This is a placeholder, real quality metrics are key
        "original_projects_ratio": 0.10,
        "activity_level": 0.10, # Increased weight for activity
        "profile_completeness": 0.05
    }
    
    weighted_score = 0.0
    total_weight = 0.0 # To normalize if some scores are missing (though current logic provides defaults)
    
    for aspect, weight in weights.items():
        score = github_aspect_scores.get(aspect, 0.0) # Default to 0 if aspect missing
        weighted_score += score * weight
        total_weight += weight
        
    # Normalize, though with current defaults, total_weight should be sum of defined weights.
    # This is more robust if aspects could be truly missing.
    # final_score = (weighted_score / total_weight) if total_weight > 0 else 0.0
    # Since all aspects are initialized, total_weight will be sum(weights.values())
    final_score = weighted_score # Assuming weights sum to 1, or this is a raw weighted sum

    return int(min(max(final_score, 0), 100))


# --- VI. Candidate Aggregation & Final Output ---

def calculate_final_candidate_score(resume_score: int, github_score: int, resume_weight: float = 0.6, github_weight: float = 0.4) -> int:
    """Combines resume and GitHub scores based on given weightage.
    Returns the final candidate score out of 100."""
    if (resume_weight + github_weight) == 0: return 0 # Avoid division by zero if weights are zero
    # Normalize weights if they don't sum to 1, though they should.
    # total_weight = resume_weight + github_weight
    # final_score = ((resume_score * resume_weight) + (github_score * github_weight)) / total_weight
    final_score = (resume_score * resume_weight) + (github_score * github_weight)
    return int(round(min(max(final_score, 0), 100))) # Ensure score is within 0-100

def process_single_candidate(
    resume_file_path: Optional[str],
    job_description: str,
    mistral_api_key: str,
    github_token: Optional[str],
    jd_expected_languages: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """Orchestrates the processing for a single candidate.
    Returns a dictionary with candidate data and scores, or None on critical failure."""
    
    if not resume_file_path or not os.path.exists(resume_file_path):
        print(f"Error: Resume file not found or path is invalid: {resume_file_path}")
        # Return a structure indicating failure for this specific path
        return {
            "resume_path": resume_file_path or "Unknown", 
            "status": "file_not_found",
            "contact_info": {}, 
            "scores": {"resume": 0, "github_overall": 0, "final": 0}, 
            "github_url": None, 
            "resume_summary": "File not found."
        }

    candidate_data: Dict[str, Any] = {
        "resume_path": resume_file_path, 
        "contact_info": {}, 
        "scores": {"resume": 0, "github_overall": 0, "final": 0}, 
        "github_url": None, 
        "resume_summary": ""
    }
    jd_langs = jd_expected_languages if jd_expected_languages else []

    print(f"Processing resume: {resume_file_path}")
    resume_text = extract_text_from_resume(resume_file_path)
    if not resume_text:
        print(f"Warning: Could not extract text from resume {resume_file_path}. Resume score will be 0.")
        candidate_data["resume_summary"] = "Could not extract text."
        # Scores already initialized to 0, so just proceed to GitHub if possible
    else:
        candidate_data["contact_info"] = extract_contact_info_from_text(resume_text)
        resume_links = extract_links_from_text(resume_text)
        
        print("Summarizing resume with Mistral...")
        resume_summary = summarize_with_mistral(resume_text, mistral_api_key)
        candidate_data["resume_summary"] = resume_summary
        
        print("Scoring resume against JD...")
        resume_score = score_resume_ats(resume_summary, job_description, mistral_api_key)
        candidate_data["scores"]["resume"] = resume_score
        
        candidate_data["github_url"] = identify_github_link(resume_links)

    github_score = 0 # Default GitHub score
    if candidate_data["github_url"]:
        github_url = candidate_data["github_url"]
        print(f"Identified GitHub URL: {github_url}")
        validation_type = validate_github_url(github_url)
        if validation_type == 'user':
            username = get_github_username_from_url(github_url)
            if username:
                print(f"Analyzing GitHub profile for user: {username}")
                github_aspect_scores = score_github_profile_aspects(
                    username, github_token, job_description, mistral_api_key, jd_langs
                )
                github_score = calculate_overall_github_score(github_aspect_scores)
                candidate_data["scores"]["github_aspects"] = github_aspect_scores
            else:
                print(f"Could not extract username from GitHub URL: {github_url}. Skipping GitHub analysis.")
        elif validation_type == 'repo':
            print(f"GitHub URL {github_url} is a repository, not a user profile. Attempting to find user from repo URL...")
            # Attempt to get username if it's like github.com/user/repo
            try:
                path_parts = [part for part in requests.utils.urlparse(github_url).path.split('/') if part]
                if len(path_parts) >= 1:
                    potential_username = path_parts[0]
                    print(f"Potential username from repo URL: {potential_username}. Analyzing this profile.")
                    github_aspect_scores = score_github_profile_aspects(
                        potential_username, github_token, job_description, mistral_api_key, jd_langs
                    )
                    github_score = calculate_overall_github_score(github_aspect_scores)
                    candidate_data["scores"]["github_aspects"] = github_aspect_scores
                else:
                    print(f"Cannot determine username from repo URL {github_url}. Skipping GitHub analysis.")
            except Exception as e_gh_parse:
                print(f"Error parsing username from repo URL {github_url}: {e_gh_parse}. Skipping GitHub analysis.")
        else:
            print(f"GitHub URL {github_url} is not identified as a user profile (type: {validation_type}). Skipping GitHub analysis.")
            
    candidate_data["scores"]["github_overall"] = github_score
    
    final_score = calculate_final_candidate_score(candidate_data["scores"]["resume"], github_score)
    candidate_data["scores"]["final"] = final_score
    
    return candidate_data


def process_all_candidates(
    resume_sources: List[str], # Can be file paths or URLs
    job_description: str,
    mistral_api_key: str,
    github_token: Optional[str],
    jd_expected_languages: Optional[List[str]] = None,
    download_dir: str = "downloaded_resumes"
) -> List[Dict[str, Any]]:
    """Processes all candidates from the given resume sources.
    Returns a list of processed candidate data dictionaries."""
    all_candidate_results = []
    
    if not os.path.exists(download_dir):
        try:
            os.makedirs(download_dir)
        except OSError as e:
            print(f"Error creating download directory {download_dir}: {e}. Downloads may fail.")
            # Continue, but downloads will likely fail if the dir can't be made.

    for i, source in enumerate(resume_sources):
        print(f"\n--- Processing candidate {i+1}/{len(resume_sources)} from source: {source} ---")
        resume_file_path = None
        
        if source.startswith("http://") or source.startswith("https://"):
            print(f"Downloading resume from URL: {source}")
            resume_file_path = download_resume(source, download_dir)
            if not resume_file_path:
                print(f"Failed to download resume from {source}. Skipping.")
                all_candidate_results.append({
                    "source": source, 
                    "status": "failed_download", 
                    "scores": {"final": 0, "resume": 0, "github_overall": 0}, 
                    "contact_info": {},
                    "resume_summary": "Failed to download."
                })
                continue
        elif os.path.isfile(source):
            resume_file_path = source
        else:
            print(f"Source '{source}' is not a valid URL or existing file path. Skipping.")
            all_candidate_results.append({
                "source": source, 
                "status": "invalid_source", 
                "scores": {"final": 0, "resume": 0, "github_overall": 0}, 
                "contact_info": {},
                "resume_summary": "Invalid source."
            })
            continue

        # resume_file_path should be valid now if we haven't continued
        candidate_result = process_single_candidate(
            resume_file_path,
            job_description,
            mistral_api_key,
            github_token,
            jd_expected_languages
        )
        
        if candidate_result: # process_single_candidate now always returns a dict
            all_candidate_results.append(candidate_result)
        # No 'else' needed as process_single_candidate is designed to return a dict even on error

    return all_candidate_results

def get_top_n_candidates(all_candidates_data: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Sorts candidates by final score and returns the top N.
    Candidates with higher scores come first."""
    if not all_candidates_data: return []
    # Ensure 'scores' and 'final' exist, defaulting to 0 if not for robust sorting
    sorted_candidates = sorted(
        all_candidates_data,
        key=lambda c: c.get("scores", {}).get("final", 0),
        reverse=True
    )
    return sorted_candidates[:n]

def display_top_candidates_info(top_candidates: List[Dict[str, Any]]):
    """Prints the scores and contact details of top candidates.
    A more sophisticated version might generate a report file (e.g., CSV, HTML)."""
    print("\n--- Top Candidates ---")
    if not top_candidates:
        print("No candidates to display or all failed processing/scoring.")
        return

    for i, candidate in enumerate(top_candidates):
        print(f"\nRank {i+1}:")
        source_display = candidate.get('resume_path', candidate.get('source', 'N/A'))
        print(f"  Source/File: {source_display}")
        
        scores = candidate.get("scores", {})
        print(f"  Final Score: {scores.get('final', 'N/A')}")
        print(f"    Resume Score: {scores.get('resume', 'N/A')}")
        print(f"    GitHub Score: {scores.get('github_overall', 'N/A')}")
        
        if "github_aspects" in scores and scores["github_aspects"]:
            print(f"    GitHub Aspects:")
            for aspect, score in scores["github_aspects"].items():
                print(f"      - {aspect.replace('_', ' ').title()}: {score:.2f}")
        
        contact = candidate.get("contact_info", {})
        print(f"  Contact:")
        print(f"    Email: {contact.get('email', 'N/A')}")
        print(f"    Phone: {contact.get('phone', 'N/A')}")
        
        if candidate.get("github_url"):
            print(f"  GitHub Profile: {candidate.get('github_url')}")
        
        summary = candidate.get('resume_summary', 'N/A')
        # print(f"  Resume Summary: {summary[:200]}..." if len(summary) > 200 else summary) # Optional
        print("-" * 20)

# --- Main Execution Example (Illustrative) ---
if __name__ == "__main__":
    # --- CONFIGURATION ---
    # IMPORTANT: Set your API keys as environment variables or directly here (not recommended for production)
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") # Optional, for higher rate limits & private data access

    if not MISTRAL_API_KEY:
        print("Error: MISTRAL_API_KEY environment variable not set. LLM features will fail.")
        # exit(1) # Or handle gracefully

    # Example Job Description
    EXAMPLE_JOB_DESCRIPTION = """
    Senior Python Developer
    We are looking for an experienced Python Developer to join our dynamic team.
    Responsibilities:
    - Develop and maintain scalable web applications using Python, Django, and Flask.
    - Work with PostgreSQL and NoSQL databases.
    - Implement RESTful APIs.
    - Utilize cloud platforms like AWS.
    - Write clean, maintainable, and testable code.
    - Collaborate with cross-functional teams.
    Required Skills:
    - 5+ years of Python development experience.
    - Strong experience with Django or Flask.
    - Proficiency in SQL (PostgreSQL) and NoSQL databases.
    - Experience with Git and CI/CD pipelines.
    - Familiarity with containerization (Docker, Kubernetes) is a plus.
    - Excellent problem-solving skills.
    - Languages: Python, JavaScript
    """
    # Expected languages from the JD (for GitHub analysis)
    JD_EXPECTED_LANGUAGES = ["Python", "JavaScript", "SQL"]


    # Example Resume Sources (mix of local files and URLs)
    # Create dummy resume files for testing if you don't have them
    # For example, create 'dummy_resume1.txt', 'dummy_resume2.pdf'
    # You'll need to have actual PDF/DOCX files or use TXT for simpler testing.
    
    # Create dummy files for testing:
    if not os.path.exists("resumes_for_testing"):
        os.makedirs("resumes_for_testing")
    
    with open("resumes_for_testing/candidate1_dev.txt", "w") as f:
        f.write("John Doe\njohn.doe@example.com\n123-456-7890\nhttps://github.com/johndoe-example\n\nExperienced Python Developer with skills in Django, Flask, and SQL. I love coding in Python and JavaScript.")
    
    # For PDF/DOCX, you'd need actual files. Let's assume you have one.
    # If you have a PDF resume, place it as "resumes_for_testing/candidate2_analyst.pdf"
    # For this example, we'll just use another TXT if PDF is not available.
    if not os.path.exists("resumes_for_testing/candidate2_analyst.pdf"):
         with open("resumes_for_testing/candidate2_analyst.txt", "w") as f:
            f.write("Jane Smith\njane.smith@example.com\n(987)654-3210\nhttps://github.com/janesmith-example\n\nData Analyst skilled in R, Python, and SQL. Project on GitHub showcasing data visualization with Python.")
            CANDIDATE2_PATH = "resumes_for_testing/candidate2_analyst.txt"
    else:
        CANDIDATE2_PATH = "resumes_for_testing/candidate2_analyst.pdf"

    EXAMPLE_RESUME_SOURCES = [
        "resumes_for_testing/candidate1_dev.txt",
        CANDIDATE2_PATH,
        "https://raw.githubusercontent.com/torvalds/linux/master/README", # Example of a non-resume URL for testing download
        "non_existent_file.docx" # To test file not found
    ]

    print("Starting candidate processing...")
    all_results = process_all_candidates(
        resume_sources=EXAMPLE_RESUME_SOURCES,
        job_description=EXAMPLE_JOB_DESCRIPTION,
        mistral_api_key=MISTRAL_API_KEY,
        github_token=GITHUB_TOKEN,
        jd_expected_languages=JD_EXPECTED_LANGUAGES,
        download_dir="downloaded_test_resumes" # Specify a download directory for URL sources
    )

    print("\n--- All Processed Candidate Data ---")
    # For brevity, just print paths and final scores here
    for res in all_results:
        status = res.get("status", "processed")
        path = res.get("resume_path", res.get("source", "N/A"))
        score = res.get("scores", {}).get("final", "N/A")
        print(f"Path: {path}, Status: {status}, Final Score: {score}")


    # Get and display top N candidates
    N_TOP_CANDIDATES = 2
    top_candidates = get_top_n_candidates(all_results, N_TOP_CANDIDATES)
    
    if top_candidates:
        display_top_candidates_info(top_candidates)
    else:
        print(f"\nNo candidates processed successfully to display top {N_TOP_CANDIDATES}.")

    print("\nProcessing finished.")
    print("Note: For full functionality, ensure ChromeDriver is in your PATH for Selenium, and all libraries are installed.")
    print("Also, ensure your MISTRAL_API_KEY is valid and has credits.")