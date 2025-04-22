from dotenv import load_dotenv
load_dotenv()
import os
import logging
import json
from groq import Groq
from kaggle.api.kaggle_api_extended import KaggleApi
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def check_environment():
    required_vars = ["GROQ_API_KEY", "KAGGLE_USERNAME", "KAGGLE_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
 
try:
    check_environment()
   
    # Initialize Groq client
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
 
    # Initialize Kaggle API
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()
    logger.info("Successfully authenticated with Kaggle API")
 
except Exception as e:
    logger.error(f"Setup error: {str(e)}")
    exit(1)
 

def get_semantic_analysis(user_query):
    """Use Groq to fully automate query understanding and search strategy"""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI expert at understanding dataset search intentions. Your task is to analyze what users are truly looking for in a dataset and extract only the essential search terms.

IMPORTANT: When users write natural language queries like "I want dataset of churn prediction" or "Looking for data about stock prices", you must extract ONLY the core search terms (e.g. "churn prediction" or "stock prices") without any of the request language. Users often phrase their needs conversationally, but the search system needs just the key terms.

Format your response as valid JSON only with no additional text."""
                },
                {
                    "role": "user",
                    "content": f"""
                    Analyze this dataset search request: '{user_query}'
                    Return a JSON object with these fields:
                    1. "search_query": ONLY the core search terms, with all request language removed (e.g., for "I want dataset of churn prediction" → "churn prediction")
                    2. "related_terms": Alternative terms or synonyms that might appear in relevant datasets
                    3. "context": The subject domain this query belongs to (e.g., 'computer science', 'finance')
                    4. "relevance_rules": Rules for determining if a dataset is relevant (e.g., "Must contain term X in title or description")
                    5. "irrelevance_signals": Terms or patterns that indicate a dataset is NOT relevant
                    """
                }
            ],
            model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=500
        )
       
        response = chat_completion.choices[0].message.content.strip()
        
        # Log the raw response for debugging
        logger.info(f"Raw Groq response: {response}")
       
        # Try to parse JSON
        try:
            analysis = json.loads(response)
            logger.info(f"Query analysis: {analysis}")
            
            # Log the extracted search query
            logger.info(f"Extracted search query: '{analysis.get('search_query', '')}'")
            
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM: {e}")
            # Extract just the search query if JSON parsing fails
            if '"search_query"' in response:
                try:
                    import re
                    search_query = re.search(r'"search_query"\s*:\s*"([^"]+)"', response)
                    if search_query:
                        query = search_query.group(1)
                        return {"search_query": query, "related_terms": [user_query],
                                "context": "general", "relevance_rules": [], "irrelevance_signals": []}
                except Exception as regex_error:
                    logger.error(f"Regex extraction error: {regex_error}")
           
            # Fallback to simple structure
            return {
                "search_query": f'"{user_query}"',
                "related_terms": [user_query],
                "context": "general",
                "relevance_rules": [],
                "irrelevance_signals": []
            }
           
    except Exception as e:
        logger.error(f"Groq API error: {e}")
        # Fallback
        return {
            "search_query": f'"{user_query}"',
            "related_terms": [user_query],
            "context": "general",
            "relevance_rules": [],
            "irrelevance_signals": []
        }

 
def evaluate_dataset_relevance(dataset, analysis):
    """Evaluate dataset relevance using the Groq-generated analysis"""
    try:
        title = getattr(dataset, "title", "").lower()
        description = getattr(dataset, "description", "").lower()
        combined = title + " " + description
       
        # Start with positive score
        score = 5
       
        # Boost for search query matches in title (most important)
        search_terms = []
        search_query = analysis.get("search_query", "")
       
        # Extract quoted phrases and individual terms
        in_quotes = False
        current_term = ""
        for char in search_query:
            if char == '"':
                in_quotes = not in_quotes
                if not in_quotes and current_term:  # End of quoted phrase
                    search_terms.append(current_term)
                    current_term = ""
            elif char == ' ' and not in_quotes:
                if current_term:  # End of unquoted term
                    search_terms.append(current_term)
                    current_term = ""
            else:
                current_term += char
       
        if current_term:  # Add any remaining term
            search_terms.append(current_term)
       
        # Clean up search terms
        search_terms = [term.strip().lower() for term in search_terms if term.strip()]
       
        # Score title matches (highest importance)
        for term in search_terms:
            if term in title:
                score += 15
       
        # Score description matches
        for term in search_terms:
            if term in description:
                score += 8
               
        # Check related terms
        for term in analysis.get("related_terms", []):
            term = term.lower()
            if term in title:
                score += 10
            elif term in description:
                score += 5
               
        # Check context/domain relevance
        context = analysis.get("context", "").lower()
        if context != "general":
            context_terms = context.split()
            for term in context_terms:
                if term in combined:
                    score += 3
                   
        # Check relevance rules (positive signals)
        for rule in analysis.get("relevance_rules", []):
            rule = rule.lower()
            if any(keyword in rule for keyword in ["title", "name"]):
                # This rule is about the title
                keywords = [word for word in rule.split() if len(word) > 3 and word not in ["must", "should", "contain", "title", "name"]]
                for keyword in keywords:
                    if keyword in title:
                        score += 8
            elif any(keyword in rule for keyword in ["description"]):
                # This rule is about the description
                keywords = [word for word in rule.split() if len(word) > 3 and word not in ["must", "should", "contain", "description"]]
                for keyword in keywords:
                    if keyword in description:
                        score += 4
            else:
                # General rule, check both
                keywords = [word for word in rule.split() if len(word) > 3 and word not in ["must", "should", "contain", "have"]]
                for keyword in keywords:
                    if keyword in combined:
                        score += 3
                       
        # Check irrelevance signals (negative signals)
        for signal in analysis.get("irrelevance_signals", []):
            signal = signal.lower()
            keywords = [word for word in signal.split() if len(word) > 3 and word not in ["not", "should", "contain", "have"]]
            for keyword in keywords:
                if keyword in title:
                    score -= 15  # Strong penalty for irrelevance signals in title
                elif keyword in description:
                    score -= 8   # Medium penalty for irrelevance signals in description
                   
        # Quality signals
        downloads = getattr(dataset, "totalDownloads", 0)
        if downloads > 10000:
            score += 5
        elif downloads > 1000:
            score += 3
           
        # Freshness bonus
        last_updated = getattr(dataset, "lastUpdated", "").lower()
        if "day" in last_updated:
            score += 3
        elif "week" in last_updated:
            score += 2
        elif "month" in last_updated:
            score += 1
           
        return score
       
    except Exception as e:
        logger.error(f"Error evaluating dataset relevance: {e}")
        return 0  # Neutral score on error
 
def search_kaggle_datasets(query, min_results=5, limit=20):
    """Search Kaggle with intelligent filtering, ensuring minimum result count"""
    try:
        # Get automated analysis of the query
        analysis = get_semantic_analysis(query)
        search_query = analysis.get("search_query", f'"{query}"')
       
        print(f"🔍 Searching Kaggle for: '{search_query}'")
       
        # First search attempt with the optimized query
        datasets = list(kaggle_api.dataset_list(
            search=search_query,
            file_type="csv",
            sort_by="hottest"
        ))
       
        logger.info(f"Found {len(datasets)} initial results")
       
        # Score datasets based on relevance
        scored_datasets = []
        for dataset in datasets:
            relevance_score = evaluate_dataset_relevance(dataset, analysis)
            scored_datasets.append((dataset, relevance_score))
           
        # Sort by relevance score
        scored_datasets.sort(key=lambda x: x[1], reverse=True)
       
        # Filter out very irrelevant datasets (negative score)
        filtered_datasets = [(ds, score) for ds, score in scored_datasets if score > 0]
       
        # If we have too few results, try to get more by expanding search
        if len(filtered_datasets) < min_results and len(analysis.get("related_terms", [])) > 0:
            print(f"⚠️ Not enough relevant results ({len(filtered_datasets)}), expanding search...")
           
            # Try with related terms
            for term in analysis.get("related_terms", [])[:3]:  # Try up to 3 related terms
                expanded_datasets = list(kaggle_api.dataset_list(
                    search=term,
                    file_type="csv",
                    sort_by="hottest"
                ))
               
                # Score and add these datasets
                for dataset in expanded_datasets:
                    if not any(dataset.ref == ds.ref for ds, _ in scored_datasets):  # Avoid duplicates
                        relevance_score = evaluate_dataset_relevance(dataset, analysis)
                        if relevance_score > 0:  # Only add somewhat relevant datasets
                            scored_datasets.append((dataset, relevance_score))
           
            # Resort the expanded list
            scored_datasets.sort(key=lambda x: x[1], reverse=True)
            filtered_datasets = [(ds, score) for ds, score in scored_datasets if score > 0]
           
        # If we still don't have enough, loosen the filtering
        if len(filtered_datasets) < min_results:
            print(f"⚠️ Still not enough results ({len(filtered_datasets)}), relaxing filters...")
            # Include all non-negative scores
            filtered_datasets = [(ds, score) for ds, score in scored_datasets if score >= -5]
       
        # Final results
        final_datasets = [ds for ds, _ in filtered_datasets[:limit]]
        logger.info(f"Returning {len(final_datasets)} filtered datasets")
       
        return final_datasets
           
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        # Fallback to simple search in case of error
        try:
            logger.info("Falling back to simple search")
            simple_datasets = list(kaggle_api.dataset_list(
                search=query,
                file_type="csv",
                sort_by="hottest",
                limit=min_results
            ))
            return simple_datasets
        except:
            return []
 
def format_size(size):
    """Format size in bytes to a human-readable format"""
    try:
        size_num = int(size) if isinstance(size, (int, str)) and str(size).isdigit() else 0
        if size_num < 1024:
            return f"{size_num} B"
        elif size_num < 1024**2:
            return f"{size_num/1024:.1f} KB"
        elif size_num < 1024**3:
            return f"{size_num/(1024**2):.1f} MB"
        else:
            return f"{size_num/(1024**3):.1f} GB"
    except:
        return str(size)
 
def download_dataset(dataset_ref, i):
    """Download a Kaggle dataset"""
    try:
        # Create downloads directory if it doesn't exist
        download_path = os.path.join(os.getcwd(), "downloads")
        if not os.path.exists(download_path):
            os.makedirs(download_path)
           
        print(f"\n⬇️ Downloading dataset #{i}: {dataset_ref} to {download_path}...")
       
        # Use the Kaggle API to download the dataset
        kaggle_api.dataset_download_files(
            dataset=dataset_ref,
            path=download_path,
            unzip=True
        )
       
        print(f"✅ Dataset #{i} downloaded successfully to {download_path}")
       
    except Exception as e:
        logger.error(f"Download error for dataset #{i}: {str(e)}")
        print(f"❌ Failed to download dataset #{i}: {str(e)}")
 
def display_datasets(results, limit=5):
    """Display datasets with detailed information and download buttons"""
    if results:
        print(f"\n📊 Found {len(results)} relevant datasets:")
       
        active_downloads = {}  # To track active downloads
       
        for i, ds in enumerate(results[:limit], 1):
            # Safely get attributes with fixed extraction for description and size
            title = getattr(ds, "title", "Untitled Dataset")
            owner = getattr(ds, "ownerName", "Unknown")
            ref = getattr(ds, "ref", "").replace("/datasets/", "")
           
            # Extract description properly
            description = getattr(ds, "description", "No description available")
            if not description or description.strip() == "":
                # Try alternate properties
                description = getattr(ds, "subtitle", "No description available")
                if not description or description.strip() == "":
                    description = getattr(ds, "summary", "No description available")
           
            # Extract size properly
            size = getattr(ds, "size", "Unknown size")
            if not size or str(size).strip() == "":
                # Try alternate properties
                size = getattr(ds, "totalBytes", "Unknown size")
               
            last_updated = getattr(ds, "lastUpdated", "Unknown")
            total_downloads = getattr(ds, "totalDownloads", 0)
           
            # Format description - truncate if too long
            desc_preview = description[:80] + "..." if len(description) > 80 else description
           
            # Format size
            size_str = format_size(size)
           
            # Print dataset information
            print(f"\n{i}. 📁 {title}")
            print(f"   👤 Owner: {owner}")
            print(f"   📝 Description: {desc_preview}")
            print(f"   📊 Size: {size_str}")
            print(f"   ⬇️  Downloads: {total_downloads}")
            print(f"   🕒 Last Updated: {last_updated}")
            print(f"   🔗 URL: https://www.kaggle.com/datasets/{ref}")
           
            # Print download button
            print(f"   [d{i}] Download this dataset")
            print("   " + "─" * 50)
           
            # Store ref in active_downloads
            active_downloads[f"d{i}"] = (ref, i)
           
        # Return the download options
        return active_downloads
    else:
        print("\n❌ No datasets found. Try a different search query.")
        return {}
 
def process_downloads(active_downloads, user_input):
    """Process download request based on user input"""
    if user_input.lower() in active_downloads:
        # Get the dataset reference and index
        dataset_ref, i = active_downloads[user_input.lower()]
       
        # Start download in a separate thread to not block the main thread
        download_thread = threading.Thread(
            target=download_dataset,
            args=(dataset_ref, i)
        )
        download_thread.daemon = True
        download_thread.start()
        return True
    return False
    

def main():
    """Main function to run the dataset search tool"""
    try:
        print("\n🔍 Smart Kaggle Dataset Search")
        print("=" * 40)
        print("This tool finds relevant datasets on Kaggle")
        print("=" * 40)
       
        active_downloads = {}  # To store available download buttons
       
        while True:
            user_input = input("\n📝 Enter dataset search or download option (or 'exit' to quit): ")
           
            # Check if it's a download request
            if active_downloads and process_downloads(active_downloads, user_input):
                continue
               
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\nThank you for using Smart Kaggle Dataset Search! Goodbye! 👋")
                break
               
            if not user_input.strip():
                print("⚠️ Please enter a search query.")
                continue
               
            logger.info(f"Starting search with query: {user_input}")
            results = search_kaggle_datasets(user_input, min_results=5)
           
            if results:
                active_downloads = display_datasets(results, 5)
               
                if len(results) > 5:
                    see_more = input("\n🔍 Do you want to see more results? (y/n): ")
                    if see_more.lower() == 'y':
                        more_downloads = display_datasets(results, min(len(results), 10))
                        # Update active downloads with the additional options
                        active_downloads.update(more_downloads)
            else:
                print("❌ No datasets found. Try a different search query.")
           
    except KeyboardInterrupt:
        print("\n\nSearch interrupted. Goodbye! 👋")
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
 
if __name__ == "__main__":
    try:
        check_environment()
        print("Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Server startup error: {str(e)}")