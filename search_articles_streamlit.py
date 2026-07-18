import requests
import streamlit as st

CORE_API_KEY = st.secrets["CORE_API_KEY"]


def search_core_papers(query=None, year_from=2010, max_results=20):
    API_URL = "https://api.core.ac.uk/v3/search/works"
    
    """
    Searches for research papers using the CORE API.
    
    :param query: String keywords to search for.
    :param year_from: Integer year to filter results from (e.g., 2023).
    :param max_results: Integer max number of papers to return.
    :return: List of paper dictionaries.
    """
    
    # Define the request payload
    payload = {
        "q": query,
        "limit": max_results,
        "fullTextOnly": True  # Only return papers with downloadable PDFs
    }
    
    # Add year filter 
    if year_from:
        payload["yearFrom"] = year_from

    # Define headers with API key
    headers = {
        "Authorization": f"Bearer {CORE_API_KEY}"
    }

    try:
        # Send the POST request
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        data = response.json()
        
        # Extract papers from the response
        # The CORE API v3 returns a 'results' key containing the list of works
        papers = data.get("results", [])
        
        return papers

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
        
    return []


def execute_search_papers(keywords):
    
    # Replace with your actual CORE API key
    #CORE_API_KEY = CORE_API_KEY
    
    # Search for papers published from 2010 onwards
    papers = search_core_papers(query=keywords, year_from=2010, max_results=20)
    
    if papers:
        st.text(f"Found {len(papers)} papers:\n")
        for paper in papers:
            title = paper.get("title", "No Title")
            year = paper.get("yearPublished", "Unknown Year")
            doi = paper.get("doi", "No DOI")
            download_url = paper.get("downloadUrl", "No Download URL")
            
            st.text(f"Title: {title}")
            st.text(f"Year: {year}")
            st.text(f"DOI: {doi}")
            st.text(f"Download: {download_url}")
            st.text("-" * 40)
    else:
        st.text("No papers found or an error occurred.")   
