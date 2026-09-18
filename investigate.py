import os
import json
import re


for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
    os.environ.pop(k, None)
os.environ["NO_PROXY"] = "*"

import ollama

## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant.\n"
        "Follow these rules strictly:\n"
        "- The model must use only the given JSON file\n"
        "- Not all the details of an item must match to be a possible match.\n"
        "- Only JSON must be returned, with exactly the following structure:\n"
        '{\n    "matches": ["ITEM_ID"],\n    "confidence": "LOW"\n}\n'
        '- "matches" contains all the possible matches (list of item IDs)\n'
        '- "confidence" measures how confident the model is about the matches. It must be exactly one of: LOW, MEDIUM, HIGH.\n'
        "- If there is no match then the model must return an empty list for matches"
    )
    
    user_prompt = (
        f"Available Items:\n{json.dumps(available_items, indent=2, ensure_ascii=False)}\n\n"
        f"User Description: {description}"
    )
    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
  
    client = ollama.Client(host="http://127.0.0.1:11434")
    response = client.chat(
        model="qwen2.5:1.5b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
        options={"temperature": 0.0},
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result. 
def parse_response(response_text):
    text = response_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


## Logic to validate the result returned by Qwen.
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if result["confidence"] not in ["LOW", "MEDIUM", "HIGH"]:
        return False

    valid_ids = {item["id"] for item in available_items if "id" in item}
    for item_id in result["matches"]:
        if not isinstance(item_id, str) or item_id not in valid_ids:
            return False

    return True


## Logic to display the matches found by Qwen in a user-friendly format.
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result.get('confidence', '')}\n")

    matches = result.get("matches", [])
    if not matches:
        print("No matches found:", matches)
        return

    print("Possible matches:\n")
    item_lookup = {item["id"]: item for item in available_items}
    for match_id in matches:
        item = item_lookup.get(match_id)
        if item:
            print(f"ID: {item.get('id')}")
            print(f"Item: {item.get('item')}")
            print(f"Color: {item.get('color')}")
            print(f"Location: {item.get('location')}")
            print(f"Date found: {item.get('date')}\n")


## Control center for the entire program.
def main():
    items_file = "found_items.json"
    output_file = "output/match_result.json"

    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50 + "\n")


    items = load_items(items_file)
    available_items = get_unclaimed_items(items)

  
    user_desc = input("Describe the item you lost: ")
    print("\nSearching for possible matches...")

  
    system_prompt, user_prompt = build_prompt(user_desc, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)


    result = parse_response(response_text)
    if not validate_result(result, available_items):
        print(f"\nModel returned an invalid response:\n{response_text}")
        return

 
    display_matches(result, available_items)
    save_result(result, output_file)
    print(f"Result saved to {output_file}")


if __name__ == "__main__":
    main()