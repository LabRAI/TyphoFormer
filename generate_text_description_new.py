"""
generate_text_description.py
--------------------------------------------------
Generate enhanced natural language descriptions for each typhoon record
using GPT-4o, with knowledge-augmented meteorological context.

Author: Lincan Li
Date: 2025-06-02
"""

import os
import time
import backoff
import pandas as pd
from openai import OpenAI

# =====================================================
# Configuration
INPUT_CSV = "HURDAT_2new_3000.csv"
OUTPUT_CSV = INPUT_CSV.replace(".csv", "_with_descriptions.csv")


API_KEY = "Your_Own_OpenAI_API_Key" #place your own OpenAI API Keys here
client = OpenAI(api_key=API_KEY)


# =====================================================
# Build Enhanced Prompt

def build_prompt(row: pd.Series) -> str:
    """
    Build a meteorologically rich GPT-4o prompt dynamically
    from one row of typhoon numerical features.
    """

    # Convert all columns in the current row into a readable "key: value" block
    record_lines = []
    for col, val in row.items():
        if pd.isna(val):
            continue  # skip missing values
        record_lines.append(f"{col}: {val}")
    record_block = "\n".join(record_lines)

    # Knowledge-augmented prompt template
    prompt = f"""
You are a senior meteorologist and AI forecasting researcher. 
You have access to both structured typhoon records and extensive meteorological knowledge from global databases.

Your task:
Given the following structured typhoon record, generate a comprehensive, natural-language description
of the typhoon’s situation at that specific time. 

You should:
1. Describe the typhoon’s current characteristics based on the data provided (position, wind, pressure, etc.).
2. Enrich the description using your broader knowledge — include relevant context such as:
   - The general phase of the typhoon (formation, intensification, peak, weakening, or landfall),
   - Recent or upcoming trajectory changes (e.g., turning northward, approaching land, dissipating),
   - Any notable impacts or comparisons with other storms, if known.
3. If this storm is historically known (e.g., Hurricane IAN, MILTON, etc.), briefly mention a factual background 
   or effect based on reliable meteorological information.
4. Write in an informative, fluent, and factual tone suitable for a scientific meteorological report.
5. Keep the output concise: 3–6 well-structured sentences.
6. If any data values are missing, skip them gracefully.

Here is the structured typhoon record:
{record_block}

Now write the natural-language description:
"""
    return prompt.strip()


# =====================================================
# Retry Wrapper for GPT API

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def generate_description(prompt: str) -> str:
    """
    Call GPT-4o to generate the description for one record.
    Automatically retries if API error occurs.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()


# =====================================================
# Main Processing Function
def main():
    print(f"Loading dataset: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} records.")

    #descriptions = []
    if "description" not in df.columns:
        df["description"] = [" "] * len(df)
    
    start_time = time.time()

    for idx, row in df.iterrows():
        prompt = build_prompt(row)
        try:
            desc = generate_description(prompt)
        except Exception as e:
            print(f"Error on row {idx}: {e}")
            desc = "ERROR"
        df.at[idx,"description"] = desc #update row by row

        # progress log
        if (idx + 1) % 5 == 0:
            print(f"Progress: {idx + 1}/{len(df)} records processed.")

        # periodically save
        if (idx + 1) % 10 == 0:
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"Saved intermediate results ({idx + 1} rows).")

        time.sleep(2)  # rate control

    # Save final output
    #df["description"] = descriptions
    df.to_csv(OUTPUT_CSV, index=False)
    duration = time.time() - start_time
    print(f"\n Completed! {len(df)} records processed in {duration/60:.2f} minutes.")
    print(f"Output file saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
