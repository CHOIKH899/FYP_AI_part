import os
import re
import time
import csv
import json
import pandas as pd
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Configuration
# =========================

POE_API_KEY_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "poe_api_key.txt"
)

if not os.path.exists(POE_API_KEY_FILE):
    raise FileNotFoundError(
        f"Cannot find Poe API key file:\n{POE_API_KEY_FILE}\n\n"
        "Please create 'poe_api_key.txt' in the same folder as this Python file."
    )

with open(POE_API_KEY_FILE, "r", encoding="utf-8") as f:
    POE_API_KEY = f.read().strip()

if not POE_API_KEY:
    raise ValueError(
        "The file 'poe_api_key.txt' is empty. Please paste your Poe API key into it."
    )

# Poe model name
MODEL_NAME = "GLM-5.3"

# Input and output CSV file names
INPUT_CSV_NAME = "urop_detail_summary_data.csv"
OUTPUT_CSV_NAME = "urop_detail_summary_data_split.csv"

# Number of parallel API workers
MAX_WORKERS = 4

# Optional delay inside each worker before/after API requests.
# With 4 workers, keep this small or zero unless rate limit happens.
REQUEST_DELAY_SECONDS = 0

# Maximum number of video script parts
MAX_PARTS = 6

# Target part size
MIN_WORDS_PER_PART = 8
TARGET_WORDS_PER_PART = 14
MAX_WORDS_PER_PART = 20

# Make paths relative to this Python file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_CSV = os.path.join(SCRIPT_DIR, INPUT_CSV_NAME)
OUTPUT_CSV = os.path.join(SCRIPT_DIR, OUTPUT_CSV_NAME)

# =========================
# Poe API client
# =========================

client = openai.OpenAI(
    api_key=POE_API_KEY,
    base_url="https://api.poe.com/v1",
)

# =========================
# Helper functions
# =========================

def safe_text(value):
    """
    Convert missing CSV values to empty strings.
    """
    if pd.isna(value):
        return ""
    return str(value).strip()


def count_words(text):
    """
    Count words in a text string.
    This counts normal English words, numbers, and words with apostrophes.
    """
    if not text:
        return 0

    words = re.findall(r"\b[\w']+\b", text)
    return len(words)


def clean_part_text(text):
    """
    Clean a generated part.
    """
    if text is None:
        return ""

    text = str(text).strip()

    # Remove surrounding quotes if the model adds them
    text = text.strip("\"'“”‘’")

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_split_prompt(title, summary):
    """
    Build the AI prompt for splitting one summary into video narration parts.
    """

    prompt = f"""
You are preparing narration lines for a short 30-second UROP project video.

Task:
Split the summary into natural spoken parts for video scenes.

Rules:
- Keep the original meaning.
- Do not add new information.
- Do not rewrite heavily unless needed for smooth splitting.
- Split at natural pauses, such as sentence boundaries, commas, or complete phrases.
- Do not split in the middle of a phrase in a way that sounds awkward.
- Each part should sound natural as one video scene narration.
- Avoid very short parts.
- Each part should usually be around {MIN_WORDS_PER_PART} to {MAX_WORDS_PER_PART} words.
- A good target is around {TARGET_WORDS_PER_PART} words per part.
- The whole video is around 30 seconds.
- Doubao video clips are 5 to 15 seconds, so each part should be suitable for one 5 to 15 second scene.
- Use at most {MAX_PARTS} parts.
- If fewer than {MAX_PARTS} parts are enough, use fewer parts.
- Do not create 1-3 second tiny fragments.
- Do not use bullet points in the part text.
- Do not include word counts.
- Return only valid JSON.

Output format:
Return a JSON array of strings.

Example output:
[
  "This project explores how generative AI is reshaping business, focusing on financial reporting and corporate communications.",
  "You'll review literature in natural language processing, help develop research hypotheses, and explore empirical settings.",
  "You may also assess analysis feasibility and assist with data collection or exploratory work.",
  "Along the way, you'll learn about generative AI's evolving impact on businesses.",
  "This suits students with a solid understanding of large language models and generative AI."
]

Project title:
{title}

Summary:
{summary}
""".strip()

    return prompt


def call_ai(prompt):
    """
    Send one prompt to Poe AI and return the generated text.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()


def extract_json_array(ai_text):
    """
    Extract a JSON array from the AI response.
    Handles cases where the model wraps JSON in markdown.
    """

    if not ai_text:
        raise ValueError("Empty AI response.")

    text = ai_text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # Try direct JSON parse first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting between first [ and last ]
    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1 and end > start:
        possible_json = text[start:end + 1]
        data = json.loads(possible_json)
        if isinstance(data, list):
            return data

    raise ValueError(f"Could not parse JSON array from AI response:\n{ai_text}")


def fallback_split_summary(summary, max_parts=6):
    """
    Fallback splitter if AI fails.
    This tries to split by sentences first, then balances long chunks.
    """

    summary = safe_text(summary)

    if not summary:
        return ["N/A"] * max_parts

    sentences = re.split(r"(?<=[.!?])\s+", summary)
    sentences = [s.strip() for s in sentences if s.strip()]

    parts = []

    for sentence in sentences:
        sentence_word_count = count_words(sentence)

        if sentence_word_count <= MAX_WORDS_PER_PART:
            parts.append(sentence)
        else:
            chunks = re.split(r"(?<=[,;:—-])\s+", sentence)
            current = ""

            for chunk in chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue

                if not current:
                    current = chunk
                else:
                    combined = current + " " + chunk
                    if count_words(combined) <= MAX_WORDS_PER_PART:
                        current = combined
                    else:
                        parts.append(current.strip())
                        current = chunk

            if current:
                parts.append(current.strip())

    while len(parts) > max_parts:
        merged = False

        for i in range(len(parts) - 1):
            combined = parts[i] + " " + parts[i + 1]
            if count_words(combined) <= MAX_WORDS_PER_PART:
                parts[i] = combined
                del parts[i + 1]
                merged = True
                break

        if not merged:
            parts[-2] = parts[-2] + " " + parts[-1]
            parts.pop()

    parts = [clean_part_text(p) for p in parts if clean_part_text(p)]

    while len(parts) < max_parts:
        parts.append("N/A")

    return parts[:max_parts]


def validate_and_fix_parts(parts, summary):
    """
    Ensure exactly MAX_PARTS outputs.
    Clean invalid values and pad with N/A.
    """

    cleaned_parts = []

    for part in parts:
        part = clean_part_text(part)

        if not part:
            continue

        part = re.sub(r"^\d+\.\s*", "", part).strip()

        if part:
            cleaned_parts.append(part)

    if len(cleaned_parts) > MAX_PARTS:
        allowed = cleaned_parts[:MAX_PARTS]
        extras = cleaned_parts[MAX_PARTS:]

        allowed[-1] = clean_part_text(allowed[-1] + " " + " ".join(extras))
        cleaned_parts = allowed

    if not cleaned_parts:
        return fallback_split_summary(summary, max_parts=MAX_PARTS)

    while len(cleaned_parts) < MAX_PARTS:
        cleaned_parts.append("N/A")

    return cleaned_parts[:MAX_PARTS]


def split_summary_with_ai(title, summary):
    """
    Split summary into natural video narration parts using AI.
    """

    prompt = build_split_prompt(title, summary)

    if REQUEST_DELAY_SECONDS > 0:
        time.sleep(REQUEST_DELAY_SECONDS)

    ai_text = call_ai(prompt)

    try:
        parts = extract_json_array(ai_text)
        parts = validate_and_fix_parts(parts, summary)
        return parts
    except Exception as e:
        print("Warning: AI split parsing failed. Using fallback splitter.")
        print("Reason:", e)
        return fallback_split_summary(summary, max_parts=MAX_PARTS)


def process_one_row(index, total_rows, row_dict):
    """
    Process one CSV row.
    This function runs inside a worker thread.
    """

    title = safe_text(row_dict.get("title", ""))
    summary = safe_text(row_dict.get("summary", ""))
    summary_word_count = row_dict.get("summary_word_count", "")

    print("\n" + "=" * 80)
    print(f"Worker processing row {index + 1}/{total_rows}")
    print("Title:", title)
    print("Summary word count:", count_words(summary))

    try:
        parts = split_summary_with_ai(title, summary)

        print(f"Finished row {index + 1}: {title}")
        for i, part in enumerate(parts, start=1):
            if part != "N/A":
                print(f"  part_{i}: {part} ({count_words(part)} words)")
            else:
                print(f"  part_{i}: N/A")

    except Exception as e:
        print(f"Error while processing row {index + 1}: {title}")
        print(e)
        parts = fallback_split_summary(summary, max_parts=MAX_PARTS)

    result = {
        "_row_index": index,
        "title": title,
        "summary": summary,
        "summary_word_count": summary_word_count,
        "part_1": parts[0],
        "part_2": parts[1],
        "part_3": parts[2],
        "part_4": parts[3],
        "part_5": parts[4],
        "part_6": parts[5],
    }

    return result


def save_results(results):
    """
    Save results sorted by original row order.
    """

    results_sorted = sorted(results, key=lambda x: x.get("_row_index", 0))

    output_df = pd.DataFrame(results_sorted)

    # Do not save internal helper column
    if "_row_index" in output_df.columns:
        output_df = output_df.drop(columns=["_row_index"])

    output_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        sep=",",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n"
    )


# =========================
# Main program
# =========================

def main():
    if not POE_API_KEY:
        raise ValueError("Please paste your real Poe API key into poe_api_key.txt.")

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(
            f"Cannot find the CSV file:\n{INPUT_CSV}\n\n"
            f"Please put '{INPUT_CSV_NAME}' in the same folder as this Python file."
        )

    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    print(f"Loaded {len(df)} summaries.")

    required_columns = {"title", "summary", "summary_word_count"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Input CSV is missing required columns: {missing_columns}"
        )

    results = []
    completed_keys = set()

    # Resume if output already exists
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV, encoding="utf-8-sig")

        if "title" in existing_df.columns and "summary" in existing_df.columns:
            for _, existing_row in existing_df.iterrows():
                key = (
                    safe_text(existing_row.get("title", "")),
                    safe_text(existing_row.get("summary", ""))
                )
                completed_keys.add(key)

            # Add row index again approximately by matching input rows
            existing_records = existing_df.to_dict("records")

            for record in existing_records:
                title = safe_text(record.get("title", ""))
                summary = safe_text(record.get("summary", ""))

                matched_index = 999999999

                for idx, input_row in df.iterrows():
                    if (
                        safe_text(input_row.get("title", "")) == title
                        and safe_text(input_row.get("summary", "")) == summary
                    ):
                        matched_index = idx
                        break

                record["_row_index"] = matched_index
                results.append(record)

            print(f"Found existing output file with {len(completed_keys)} completed rows.")
            print("The script will skip rows that already have split parts.")

    tasks = []

    total_rows = len(df)

    for index, row in df.iterrows():
        title = safe_text(row.get("title", ""))
        summary = safe_text(row.get("summary", ""))

        key = (title, summary)

        if key in completed_keys:
            print(f"Skipping already completed row {index + 1}/{total_rows}: {title}")
            continue

        tasks.append((index, row.to_dict()))

    print("\n" + "=" * 80)
    print(f"Rows to process: {len(tasks)}")
    print(f"Parallel workers: {MAX_WORKERS}")
    print("=" * 80)

    if not tasks:
        print("No new rows to process.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {}

        for index, row_dict in tasks:
            future = executor.submit(
                process_one_row,
                index,
                total_rows,
                row_dict
            )
            future_to_index[future] = index

        completed_count = 0

        for future in as_completed(future_to_index):
            index = future_to_index[future]

            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                save_results(results)

                print("\n" + "-" * 80)
                print(f"Saved progress after completed task {completed_count}/{len(tasks)}")
                print(f"Output file:\n{OUTPUT_CSV}")
                print("-" * 80)

            except Exception as e:
                print(f"Unexpected error in worker for row {index + 1}:")
                print(e)

    print("\n" + "=" * 80)
    print("All done.")
    print(f"Final split summary data saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()