import requests
import time
import xml.etree.ElementTree as ET

url = "https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fai4eda.github.io%2F&countColor=%23263759"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
}



def request_url():
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        print("Request successful.")
        # Parse the SVG content
        svg = ET.fromstring(response.content)
        # Find the <title> element and print its text
        title_element = svg.find('{http://www.w3.org/2000/svg}title')
        if title_element is not None:
            total_visitors = title_element.text.replace("visitors: ","")
            total_visitors = int(total_visitors)
            return total_visitors
        else:
            print("Title element not found.")
            raise Exception("Title element not found.")
    else:
        print(f"Request failed with status code: {response.status_code}")
        raise Exception(f"Request failed with status code: {response.status_code}")

while True:
    total_visitors = request_url()
    print("Total visitors:", total_visitors)
    time.sleep(0.01)
    if total_visitors > 12219:
        break
