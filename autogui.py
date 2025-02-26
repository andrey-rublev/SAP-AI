import pyautogui
import pytesseract
import numpy as np
import cv2
from PIL import Image

# Set up Tesseract for OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to capture the screen (specific region for game)
def capture_screen(region=None):
    screenshot = pyautogui.screenshot(region=region)
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

# Function to extract text using OCR
def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

# Function to simulate a mouse click at the given coordinates
def click_at(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()

# Define the actions to be performed in the Arena (select pet, roll, start battle)
def perform_action(action):
    if action == 0:  # Select Pet
        click_at(500, 500)
        click_at(500, 500)  # Adjust with correct coordinates
    elif action == 1:  # Roll Shop
        click_at(600, 500)  # Adjust with correct coordinates
    elif action == 2:  # Start Battle
        click_at(700, 500)  # Adjust with correct coordinates
