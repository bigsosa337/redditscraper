import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

def get_reddit_comments(url, max_comments=100):
    # Set up Selenium options
    options = Options()
    # options.add_argument("--headless")  # Uncomment for headless mode

    # Set up the WebDriver (assume GeckoDriver is in the PATH)
    service = Service('./geckodriver.exe')  # Update with the actual path
    driver = webdriver.Firefox(service=service, options=options)

    print("Opening Reddit URL...")
    driver.get(url)
    time.sleep(5)  # Let the page load

    # Initialize last_height for the first comparison
    last_height = driver.execute_script("return document.body.scrollHeight")

    # Scroll and load comments
    comments = []
    scroll_attempts = 0
    while len(comments) < max_comments and scroll_attempts < 10:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Wait for comments to load

        # Check for and click the "View more comments" button if it exists
        try:
            load_more_button = driver.find_element(By.XPATH, "//button[contains(@class, 'button-small') and contains(@class, 'button-brand') and contains(@class, 'items-center') and contains(@class, 'justify-center')]")
            if load_more_button:
                load_more_button.click()
                print("Clicked 'View more comments' button.")
                time.sleep(3)  # Wait for more comments to load
        except Exception as e:
            print("No 'View more comments' button found or not clickable.")

        # Parse the page with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        comment_elements = soup.find_all('div', {'id': '-post-rtjson-content'})

        new_comments = 0
        for element in comment_elements:
            p_tags = element.find_all('p')
            for p in p_tags:
                comment_text = p.get_text().strip()
                if comment_text and comment_text not in comments:
                    comments.append(comment_text)
                    new_comments += 1

        print(f"Found {new_comments} new comments. Total comments so far: {len(comments)}")

        # Check if we've reached the bottom of the page
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            scroll_attempts += 1
        else:
            scroll_attempts = 0
        last_height = new_height

        # Break if maximum number of comments is reached
        if len(comments) >= max_comments:
            break

    driver.quit()
    return comments[:max_comments]

# Example usage
url = 'https://www.reddit.com/r/gaming/comments/1d8ueuh/iron_giant_player_has_character_removed_from_game/'
comments = get_reddit_comments(url, max_comments=100)

# Save comments to a CSV file
df = pd.DataFrame(comments, columns=['Comment'])
df.to_csv('reddit_comments.csv', index=False)
print(f"Scraped {len(comments)} comments and saved to reddit_comments.csv")
