import requests


def send_request(sentences):
    url = "http://127.0.0.1:5000/query"
    try:
        response = requests.post(url, json={"sentences": sentences})
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    sentences = ["Hello, world!", "This is a test sentence."]
    result = send_request(sentences)
    print(result)
