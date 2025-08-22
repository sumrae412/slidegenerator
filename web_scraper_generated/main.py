import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
import logging
from typing import List, Tuple, Optional
import time
import random

class EcommerceScraper:
    """
    A web scraper class for extracting product information from e-commerce websites.
    
    Attributes:
        base_url (str): The base URL of the e-commerce website
        headers (dict): HTTP headers to use for requests
    """
    
    def __init__(self, base_url: str):
        """
        Initialize the scraper with a base URL and default headers.
        
        Args:
            base_url (str): The base URL of the e-commerce website
        """
        self.base_url = base_url
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Set up logging
        logging.basicConfig(
            filename='scraper.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def get_page_content(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch and parse the HTML content of a given URL.
        
        Args:
            url (str): The URL to scrape
            
        Returns:
            BeautifulSoup: Parsed HTML content or None if request fails
        """
        try:
            # Add random delay to avoid overwhelming the server
            time.sleep(random.uniform(1, 3))
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            return BeautifulSoup(response.content, 'html.parser')
        
        except requests.RequestException as e:
            logging.error(f"Error fetching URL {url}: {str(e)}")
            return None
            
    def extract_product_info(self, soup: BeautifulSoup) -> List[Tuple[str, float]]:
        """
        Extract product titles and prices from the parsed HTML.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            List[Tuple[str, float]]: List of (product_title, price) tuples
        """
        products = []
        
        try:
            # Note: These selectors need to be adjusted based on the specific website's HTML structure
            product_containers = soup.find_all('div', class_='product-container')
            
            for container in product_containers:
                try:
                    title = container.find('h2', class_='product-title').text.strip()
                    price_text = container.find('span', class_='price').text.strip()
                    
                    # Convert price text to float (remove currency symbol and convert to float)
                    price = float(price_text.replace('$', '').replace(',', ''))
                    
                    products.append((title, price))
                    
                except (AttributeError, ValueError) as e:
                    logging.warning(f"Error extracting product info: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error processing HTML: {str(e)}")
            
        return products
    
    def export_to_csv(self, products: List[Tuple[str, float]], filename: str = None) -> bool:
        """
        Export the scraped product information to a CSV file.
        
        Args:
            products (List[Tuple[str, float]]): List of product information
            filename (str, optional): Output filename. Defaults to timestamp-based filename.
            
        Returns:
            bool: True if export successful, False otherwise
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'products_{timestamp}.csv'
            
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Product Title', 'Price'])
                writer.writerows(products)
                
            logging.info(f"Successfully exported {len(products)} products to {filename}")
            return True
            
        except Exception as e:
            logging.error(f"Error exporting to CSV: {str(e)}")
            return False
            
    def scrape_products(self, num_pages: int = 1) -> List[Tuple[str, float]]:
        """
        Scrape products from multiple pages.
        
        Args:
            num_pages (int): Number of pages to scrape
            
        Returns:
            List[Tuple[str, float]]: Combined list of products from all pages
        """
        all_products = []
        
        for page in range(1, num_pages + 1):
            url = f"{self.base_url}/page/{page}"
            logging.info(f"Scraping page {page}: {url}")
            
            soup = self.get_page_content(url)
            if soup:
                products = self.extract_product_info(soup)
                all_products.extend(products)
            
        return all_products

def main():
    """
    Example usage of the EcommerceScraper class.
    """
    # Example usage with a fictional e-commerce site
    scraper = EcommerceScraper('https://example-ecommerce.com')
    
    # Scrape 3 pages of products
    products = scraper.scrape_products(num_pages=3)
    
    # Export results to CSV
    if products:
        scraper.export_to_csv(products)
        print(f"Successfully scraped {len(products)} products")
    else:
        print("No products were scraped")

if __name__ == "__main__":
    main()