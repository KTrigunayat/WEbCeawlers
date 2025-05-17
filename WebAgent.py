import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

def analyze_github_repo(repo_url):
    """
    Simulates accessing a GitHub repository page and extracts key information.

    Args:
        repo_url (str): The URL of the public GitHub repository.

    Returns:
        dict: A dictionary containing extracted repository details,
              or None if access fails or data cannot be extracted.
    """
    print(f"Agent: Accessing repository URL: {repo_url}")

    headers = {
        'User-Agent': 'GitHub Hiring Evaluation Agent/1.0 (+https://example.com/hiring-agent)'
    }

    try:
        response = requests.get(repo_url, headers=headers, timeout=10)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Agent: Error accessing URL {repo_url}: {e}")
        return None

    print("Agent: Successfully accessed the page. Parsing HTML...")
    soup = BeautifulSoup(response.content, 'html.parser')

    repo_data = {}

    try:
        # Repository Name
        name_element = soup.find('strong', itemprop='name')
        repo_data['name'] = name_element.text.strip() if name_element else 'N/A'

        # Repository Description
        description_element = soup.find('p', class_='f4 my-3')
        repo_data['description'] = description_element.text.strip() if description_element else 'No description provided'

        # Stars, Forks, Watchers (often in js-social-count or similar counters)
        repo_data['stars'] = 0
        repo_data['forks'] = 0
        # repo_data['watchers'] = 0 # Watchers are tricky to parse reliably

        # Look for common locations of star/fork counts
        star_link = soup.find('a', href=re.compile(r'/stargazers'))
        if star_link:
             count_span = star_link.find('span', class_='Counter') or star_link.find('strong') # Sometimes count is strong
             if count_span:
                 count_text = count_span.text.strip().replace(',', '')
                 if count_text.isdigit():
                     repo_data['stars'] = int(count_text)

        fork_link = soup.find('a', href=re.compile(r'/network/members'))
        if fork_link:
            count_span = fork_link.find('span', class_='Counter') or fork_link.find('strong') # Sometimes count is strong
            if count_span:
                count_text = count_span.text.strip().replace(',', '')
                if count_text.isdigit():
                     repo_data['forks'] = int(count_text)

        # Primary Language(s)
        language_elements = soup.select('.color-fg-default.text-bold.mr-1')
        languages = [lang.text.strip() for lang in language_elements]
        repo_data['languages'] = languages if languages else ['Not specified or detected']

        # Commit Count
        commits_link = soup.find('a', href=re.compile(r'/commits/'))
        if commits_link:
            # Try finding the count number within the link or nearby
            commit_count_element = commits_link.find('strong') or commits_link # Sometimes strong, sometimes part of the text
            if commit_count_element:
                commit_count_text = commit_count_element.get_text(strip=True)
                match = re.search(r'(\d+)', commit_count_text)
                if match:
                    repo_data['commit_count'] = int(match.group(1).replace(',', ''))
                else:
                     repo_data['commit_count'] = 'N/A (Count text not parsed)'
            else:
                 repo_data['commit_count'] = 'N/A (Count element not found)'
        else:
            repo_data['commit_count'] = 'N/A (Commits link not found)'


        # Last Updated / Last Commit Date
        last_updated_element = soup.find('relative-time')
        repo_data['last_updated'] = last_updated_element.text.strip() if last_updated_element else 'N/A (Relative Time)'

        # More exact last commit time from datetime attribute if available
        last_commit_time_element = soup.find('span', class_='react-wrap-last-commit-time')
        if last_commit_time_element and 'datetime' in last_commit_time_element.attrs:
            repo_data['last_updated_exact'] = last_commit_time_element['datetime']
        else:
             repo_data['last_updated_exact'] = 'N/A (Exact Time)'


        # License Information
        # Look for common elements indicating license
        license_element = soup.find('a', href=re.compile(r'/blob/.*/LICENSE', re.IGNORECASE)) # Common link pattern
        if not license_element:
             # Try finding it in a sidebar info box link
             info_box = soup.find('div', class_='Box-body')
             if info_box: # Check if info_box was found before searching within it
                 license_element = info_box.find('a', title=re.compile(r'license', re.IGNORECASE))

        if license_element:
             repo_data['license'] = license_element.get_text(strip=True)
        else:
             repo_data['license'] = 'No license found'

        # README presence (Check for the rendered README section header)
        readme_header = soup.find('h2', string=lambda t: t and 'README' in t)
        repo_data['has_readme'] = bool(readme_header)

        # Contributors (Link to contributors page)
        contributors_link = soup.find('a', href=re.compile(r'/graphs/contributors'))
        if contributors_link:
            # Number is often in a span or strong within the link
            contributor_count_element = contributors_link.find('span', class_='Counter') or contributors_link.find('strong')
            if contributor_count_element:
                contributor_count_text = contributor_count_element.get_text(strip=True)
                match = re.search(r'(\d+)', contributor_count_text)
                if match:
                     repo_data['contributors_count'] = int(match.group(1).replace(',', ''))
                else:
                     repo_data['contributors_count'] = 'N/A (Count text not parsed)'
            else:
                 repo_data['contributors_count'] = 'N/A (Count element not found)'
        else:
            repo_data['contributors_count'] = 'N/A (Contributors link not found)'


    except Exception as e:
        # Catching a broader exception during parsing to see what went wrong
        print(f"Agent: Unexpected error during data extraction: {e}")
        import traceback
        traceback.print_exc() # Print traceback for debugging
        repo_data['error'] = f'Extraction error: {e}'
        # Continue and return partial data
        return repo_data

    print("Agent: Data extraction complete.")
    return repo_data

# Keep the evaluate_repo_for_hiring function the same
def evaluate_repo_for_hiring(repo_data):
    # ... (same as before)
    if not repo_data or 'error' in repo_data:
        return f"Evaluation failed: Could not retrieve or parse repository data. {'Error: ' + repo_data.get('error') if repo_data else ''}"

    evaluation = []
    name = repo_data.get('name', 'Unknown Repository')
    evaluation.append(f"--- Hiring Evaluation Summary for: {name} ---")

    # Technical Proficiency (Inferred)
    languages = repo_data.get('languages', [])
    evaluation.append(f"\nTechnical Proficiency:")
    if languages and languages != ['Not specified or detected']:
         evaluation.append(f"- Primary Language(s): {', '.join(languages)}")
         # Could add checks here for specific desired languages
    else:
         evaluation.append("- Primary language not clearly identified or project is multi-language/other.")

    # Code Quality & Maintainability (Inferred)
    evaluation.append(f"\nCode Quality & Maintainability (Inferred):")
    evaluation.append(f"- Description: {'Present' if repo_data.get('description') not in ['No description provided', 'N/A'] else 'Missing/Brief'}. Description: '{repo_data.get('description', '')[:100]}{'...' if len(repo_data.get('description', '')) > 100 else ''}'") # Handle potential None/N/A for slicing
    evaluation.append(f"- README: {'Present' if repo_data.get('has_readme') else 'Missing'}. README is crucial for project understanding.")
    evaluation.append(f"- License: {'Found' if repo_data.get('license') != 'No license found' else 'Missing'}. License: '{repo_data.get('license')}'")

    # Activity & Dedication
    evaluation.append(f"\nActivity & Dedication:")
    commit_count = repo_data.get('commit_count', 'N/A')
    evaluation.append(f"- Commit Count: {commit_count}")

    last_updated = repo_data.get('last_updated', 'N/A (Relative Time)')
    last_updated_exact_str = repo_data.get('last_updated_exact', 'N/A (Exact Time)')

    evaluation.append(f"- Last Updated (Relative): {last_updated}")
    evaluation.append(f"- Last Updated (Exact): {last_updated_exact_str}")

    if last_updated_exact_str not in ['N/A', 'N/A (Exact Time)']:
        try:
            # Ensure the datetime string is in a parseable format (e.g., ISO 8601)
            # Handle 'Z' by replacing with '+00:00' for timezone awareness
            parseable_dt_str = last_updated_exact_str.replace('Z', '+00:00')
            last_updated_dt = datetime.fromisoformat(parseable_dt_str)
            now = datetime.now().astimezone(last_updated_dt.tzinfo if last_updated_dt.tzinfo else None) # Make 'now' timezone aware if possible
            delta = now - last_updated_dt

            if delta < timedelta(days=90):
                evaluation.append("- Activity Level: Recent and active.")
            elif delta < timedelta(days=365):
                 evaluation.append("- Activity Level: Moderately active within the last year.")
            else:
                 evaluation.append("- Activity Level: Low or inactive recently (last commit over a year ago).")
        except ValueError:
             evaluation.append("- Activity Level: Cannot determine exact recency due to parsing error.")
        except Exception as e:
             evaluation.append(f"- Activity Level: Cannot determine recency due to error: {e}")

    else:
        evaluation.append("- Activity Level: Cannot determine recency.")


    # Collaboration & Communication (Inferred)
    evaluation.append(f"\nCollaboration & Communication (Inferred):")
    contributors_count = repo_data.get('contributors_count', 'N/A')
    evaluation.append(f"- Contributors: {contributors_count}. {'Suggests potential collaboration experience.' if isinstance(contributors_count, int) and contributors_count > 1 else 'Likely a personal project or single contributor.'}")
    # Note: Actual collaboration requires looking at issues/PRs, which is complex parsing

    # Project Impact/Interest
    evaluation.append(f"\nProject Impact/Interest:")
    stars = repo_data.get('stars', 0)
    forks = repo_data.get('forks', 0)
    evaluation.append(f"- Stars: {stars}. {'Indicative of external interest/usefulness.' if stars > 10 else 'Low external interest observed.'}")
    evaluation.append(f"- Forks: {forks}. {'Indicates others have built upon the project.' if forks > 5 else 'Low number of forks observed.'}")

    # Relevance (Requires external context - placeholder)
    evaluation.append(f"\nRelevance to Role:")
    evaluation.append("- Assessment of relevance requires knowing the specific role requirements.")


    evaluation.append("\n--- End of Evaluation ---")

    return "\n".join(evaluation)


# Keep the main execution block the same
if __name__ == "__main__":
    # Example Usage: Replace with the actual GitHub URL you want to analyze
    github_url = input("Enter the GitHub repository URL (e.g., https://github.com/user/repo): ")

    # Basic URL validation
    if not github_url.startswith('https://github.com/'):
        print("Agent: Invalid GitHub URL format. Please provide a URL starting with 'https://github.com/'.")
    else:
        repo_info = analyze_github_repo(github_url)

        if repo_info:
            # Print extracted raw data (optional for debugging)
            # print("\n--- Extracted Raw Data ---")
            # for key, value in repo_info.items():
            #     print(f"{key}: {value}")
            # print("------------------------")

            # Perform hiring evaluation
            evaluation_summary = evaluate_repo_for_hiring(repo_info)
            print("\n" + evaluation_summary)
        else:
            print("\nAgent: Analysis failed completely. Could not access or parse.")