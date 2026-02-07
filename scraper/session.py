"""HTTP session helpers for interacting with the Senate eFD search site.

This module is responsible for creating a ``requests.Session`` that
behaves similarly to a real browser visit to:

    https://efdsearch.senate.gov/search/

It loads the search page to obtain cookies (including the CSRF token)
which are then used for subsequent AJAX POST requests.
"""

from __future__ import annotations

from typing import Tuple

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://efdsearch.senate.gov"
SEARCH_URL = f"{BASE_URL}/search/"


def create_efd_session():
    session = requests.Session()
    
    # 1. Use a real Browser User-Agent
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://efdsearch.senate.gov/search/home/'
    })

    # 2. Visit the home page to get the CSRF token
    home_url = "https://efdsearch.senate.gov/search/home/"
    response = session.get(home_url)
    
    # 3. Extract the CSRF token from the page
    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_token = soup.find('input', {'name': 'csrfmiddlewaretoken'})
    
    if csrf_token:
        # 4. "Click" the agreement button
        #    This must mirror the real browser form POST, which sends
        #    `prohibition_agreement=1` plus the CSRF token.
        payload = {
            'csrfmiddlewaretoken': csrf_token['value'],
            'prohibition_agreement': '1',
        }
        session.post(home_url, data=payload)
    
    return session, None


def default_ajax_headers(csrf_token: str) -> dict:
    """Returns headers that mirror the browser's AJAX request."""
    return {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "x-csrftoken": csrf_token,
        "x-requested-with": "XMLHttpRequest",
        "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "referrer": "https://efdsearch.senate.gov/search/",
    }