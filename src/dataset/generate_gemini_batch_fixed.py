import time
import sys
import json
import os
from google import genai
from google.genai import types
from pydantic import BaseModel
from src.dataset.uno_bench_loader import UNOBenchLoader

client = genai.Client()

completed_states = set([
     'JOB_STATE_SUCCEEDED',
     'JOB_STATE_FAILED',
     'JOB_STATE_CANCELLED',
     'JOB_STATE_EXPIRED',
 ])

def get_candidate_text(candidate, thought=False):
    parts = candidate.get("content", {}).get("parts", [])
    texts = []
    for part in parts:
        is_thought = part.get("thought", False)
        if thought and is_thought:
            texts.append(part.get("text", ""))
        elif not thought and not is_thought:
            texts.append(part.get("text", ""))
    return "".join(texts)

def get_parsed(response, sample):

    candidates = response.get("candidates", [])
    candidate = candidates[0] if candidates else {}

    thought_text = get_candidate_text(candidate, thought=True)
    answer_text  = get_candidate_text(candidate, thought=False)
    ground_truth = getattr(sample, "answer", "")

    assert ground_truth, "could not find ground truth!"
    
    if not thought_text:
        prompt = f"""
You are given the reasoning process together with final answer of a model, along with a ground truth answer.

REASONING and FINAL ANSWER:
{answer_text}

GROUND TRUTH:
{ground_truth}

Your tasks:
1. Parse the reasoning above into clear, logical steps. The last step must include only the final answer.
2. Compare the final answer to the ground truth and determine if they are equivalent.

Return a JSON object with:
- "steps": a list of strings, each representing one logical step from the reasoning
- "correct": a boolean, true if the final answer matches the ground truth, false otherwise
"""
    else:
        prompt = f"""
You are given the reasoning process and final answer of a model, along with a ground truth answer.

REASONING:
{thought_text}

FINAL ANSWER:
{answer_text}

GROUND TRUTH:
{ground_truth}

Your tasks:
1. Parse the reasoning above into clear, logical steps.
2. Compare the final answer to the ground truth and determine if they are equivalent.

Return a JSON object with:
- "steps": a list of strings, each representing one logical step from the reasoning
- "correct": a boolean, true if the final answer matches the ground truth, false otherwise
"""

    #parsed_schema = types.Schema(
    #    type=types.Type.OBJECT,
    #    properties={
    #        "steps": types.Schema(
    #            type=types.Type.ARRAY,
    #            items=types.Schema(type=types.Type.STRING)
    #        ),
    #        "correct": types.Schema(
    #            type=types.Type.BOOLEAN
    #        )
    #    },
    #    required=["steps", "correct"]
    #)
    
    # Define your schema as a Pydantic model
    class StepResponse(BaseModel):
        steps: list[str]
        correct: bool

    # FIX #2: Updated to a valid model name. Replace with whichever Gemini
    # model you have access to (e.g. "gemini-2.5-pro-preview-03-25").
    gemini_response = client.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            #response_schema=parsed_schema,
            response_schema=StepResponse
        )
    )

    # FIX #5: Guard against None before returning
    parsed = gemini_response.parsed
    if parsed is None:
        raise ValueError(
            "Structured output parsing returned None. "
            "The model may not have returned valid JSON matching the schema."
        )

    return parsed

def transform(sample, response, key):
    candidates = response.get("candidates", [])

    parsed = get_parsed(response, sample)

    # There is one candidate (index 0) whose parts are split by
    # thought=True (the reasoning trace) and thought=False (the final answer).
    candidate = candidates[0] if candidates else {}
    thought_text = get_candidate_text(candidate, thought=True)
    if not thought_text:
        thought_text_answer = parsed.steps
        thought_text = thought_text_answer[:-1]
        answer_text = thought_text_answer[-1]
    else:
        answer_text  = get_candidate_text(candidate, thought=False)

    # FIX #6a: Read finish_reason from the candidate field ("finishReason")
    # rather than hardcoding "stop". Lowercase for normalised storage.
    finish_reason = candidate.get("finishReason", "UNKNOWN").lower()

    # FIX #6b: Read model version from the response field ("modelVersion")
    # rather than hardcoding a string that may not match the job's actual model.
    model_version = response.get("modelVersion", "unknown")
    
    thought_text = " ".join(thought_text)

    result = {
        "metadata": {
            "temperature":        1.0,
            "top_p":              0.7,
            "num_images":         len(getattr(sample, "image_paths", None) or []),
            "num_audios":         len(getattr(sample, "audio_paths", None) or []),
            "num_videos":         len(getattr(sample, "video_paths", None) or []),
            "question":           getattr(sample, "question", ""),
            "finish_reason":      finish_reason,
            "model":              model_version,
            "api":                "gemini",
            "total_tokens":       response.get("usageMetadata", {}).get("totalTokenCount", 0),
            "completion_index":   candidate.get("index", 0),
            "original_idx":       int(key),
            "correct":            parsed.correct,
        },
        "final_answer": answer_text,
        "text":         thought_text + answer_text,
        "steps":        parsed.steps,
        "log_probs":    None,
    }

    return result

# FIX #2: Updated default model name to a valid Gemini model.
# Replace with whichever flash/lite model you have access to.
def generate_batch(input_file, n, model_name="gemini-3.1-flash-lite-preview"):
    filename = input_file[:input_file.rfind(".")] + "_" + model_name + "_" + str(n)

    uploaded_file = client.files.upload(
            file=input_file,
            config=types.UploadFileConfig(display_name=filename, mime_type='jsonl')
    )

    file_batch_job = client.batches.create(
        model=model_name,
        src=uploaded_file.name,
        config={
            'display_name': model_name + "_" + filename,
        },
    )

    job_name = file_batch_job.name

    print(f"Polling status for job: {job_name}")
    batch_job = client.batches.get(name=job_name)
    while batch_job.state.name not in completed_states:
        print(f"Current state: {batch_job.state.name}")
        time.sleep(30)
        batch_job = client.batches.get(name=job_name)

    # FIX #7: Check terminal state before attempting download
    if batch_job.state.name != 'JOB_STATE_SUCCEEDED':
        raise RuntimeError(
            f"Batch job '{job_name}' ended with non-success state: {batch_job.state.name}"
        )

    content_bytes = client.files.download(file=batch_job.dest.file_name)

    full_filename = filename + ".jsonl"
    with open(full_filename, "wb") as f:
        f.write(content_bytes)

    return full_filename

def combine_n_verify_outputs(input_filenames):

    samples = UNOBenchLoader("/scratch/datasets/uno-bench")

    out_filename = input_filenames[0][:input_filenames[0].rfind(".")] + "_complete.json"

    # FIX #3: len() was used as an iterable; use the line count to size the list
    with open(input_filenames[0]) as f:
        combined_results = [None] * len(f.readlines())

    for input_filename in input_filenames:
        with open(input_filename) as f:
            for line in f.readlines():
                dline = json.loads(line)
                # FIX #4: Capture the key before transform() discards it
                key = int(dline["key"])
                sample = samples[key]
                dline = transform(sample, dline["response"], key)
                if combined_results[key] is None:
                    combined_results[key] = [dline]
                else:
                    combined_results[key].append(dline)

    with open(out_filename, "w") as f:
        json.dump(combined_results, f)

if __name__ == "__main__":
    # TODO: CHECK THOROUGHLY THIS CODE (PARTICULARLY, WE COULD DIVIDE THE GENERATION AND COMBINE AND VERIFY FUNCTIONS FOR SAFETY)
    import argparse

    parser = argparse.ArgumentParser(description="Process input file and candidate count.")

    parser.add_argument("--input_file", type=str, help="Path to the input file")
    parser.add_argument("--n_candidates", type=int, help="Number of candidates")
    parser.add_argument(
        "--start-from-idx",
        type=int,
        default=0,
        help="Optional index to start the analysis from (default: 0)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-3.1-flash-lite-preview",
        help="Optional model name to use among the available ones with Gemini API (default gemini 3.1 flash lite)"
    )

    args = parser.parse_args()

    input_file = args.input_file
    n_candidates = args.n_candidates
    start_from_idx = args.start_from_idx
    model_name = args.model_name

    out_files = []
    for n in range(n_candidates):
        if n>=start_from_idx:
            out_files.append(generate_batch(input_file, n, model_name=model_name))
        else:
            # Make sure previous files were already generated
            filename = input_file[:input_file.rfind(".")] + "_" + model_name + "_" + str(n) + ".jsonl"
            assert os.path.exists(filename), f"Previous file {filename} was not generated yet: did you include the correct start index?"
            print(f"The following file already exists: {filename}. Skipping its creation...")
            out_files.append(filename)
    print(" ".join(out_files))
    combine_n_verify_outputs(out_files)
