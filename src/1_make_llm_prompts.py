from typing import List
import os
import transformers
import torch
from omegaconf import OmegaConf
from typing import List, Optional
from dataclasses import dataclass, field
import json

from yaml import warnings

ACCESS_TOKEN = None # Add your huggingface access token here

SYSTEM_PROMPT =  """
As an AI language model assistant, your task is to provide descriptive captions for images showing spurious features.

A spurious feature is a visual element that frequently co-occurs with a given object in images and may cause AI models to incorrectly recognize the object, even when it is not present.

Task Overview:

You will be given:
- An object.

Your job is to:

1. Think of potential spurious features: Identify objects, scenes, or elements that frequently co-occur with the given object in images. These should not include any parts or components of the object itself.

2. Generate 50 unique and diverse prompts describing images that contain only these spurious features, without including the object itself or any of its parts.

Important Guidelines:

- Do Not Mention the Object Name or Any Part of It: Avoid any direct or indirect references to the object name. If the object name is a composite or compound word, do not include any part of the object name in the prompts. For example, if the object is "firetruck," do not use "fire" or "truck" in the prompts.

- Do Not Mention Parts of the Object: Do not include any parts or components of the object in the prompts. For example, if the object is "mountainbike," do not use "handlebar," "gear shift," or "saddle" in the prompts.

- Do Not Include the Object Name in Written Text: Do not create prompts that refer to written text containing the object name or any part of it. For example, avoid descriptions like "a sign that says 'hummingbird'."

- Focus on Spurious Features: Use features that are likely correlated with the object due to frequent co-occurrence in images.

- Combining Elements: You may combine elements if they logically make sense to appear together in one image. Do not combine elements unlikely to co-occur.

- Ensure Diversity: Each prompt should be unique and cover different aspects of the spurious features.

- Avoid Repetition: Do not repeat prompts or make minor variations of the same prompt.

- Style and Detail: Write clear, creative, and descriptive prompts. Keep each prompt concise.

- Language and Grammar: Use proper grammar and spelling.

- Content Restrictions: Do not include offensive, sensitive, or inappropriate content.

- Avoid Bias: Ensure prompts are inclusive and free from cultural, gender, or racial bias.

- Verification: Before submitting, review the prompts to ensure they comply with all guidelines.

Examples:

For the object "hummingbird":

- Correct Prompts:
  - "Close-up of a bird feeder hanging in a lush garden."
  - "A garden filled with vibrant red flowers."
  - "Green foliage glistening after a rainfall."
  - "A bird feeder surrounded by blooming plants."
  - "Red tubular flowers swaying in the breeze."

- Incorrect Prompts (Do Not Use):
  - "A hummingbird hovering near a flower."
  - "Close-up of a hummingbird's wings in motion."
  - "A small bird with iridescent feathers perched on a branch."
  - "A sign with the word 'hummingbird' in a botanical garden."

For the object "firetruck":

- Correct Prompts:
  - "A fire station with bright red doors."
  - "Close-up of a spinning emergency siren light."
  - "Firefighters conducting a training drill."
  - "A tall ladder reaching up the side of a building."
  - "Protective gear hanging neatly in a station locker room."

- Incorrect Prompts (Do Not Use):
  - "A bright red firetruck parked on the street."
  - "Children waving at a passing firetruck."
  - "A sign that reads 'Fire Station No. 1'."
  - "A red truck with emergency equipment."
  - Using the words "fire" or "truck" in the prompts.

For the object "mountainbike":

- Correct Prompts:
  - "A winding trail cutting through a dense forest."
  - "A helmet resting on a tree stump beside a path."
  - "Sunlight filtering through trees along a forest trail."
  - "A backpack leaning against a wooden signpost on a hillside."
  - "A group of friends hiking through mountainous terrain."

- Incorrect Prompts (Do Not Use):
  - "A mountainbike leaning against a tree."
  - "Close-up of a mountainbike's gears."
  - "A cyclist adjusting the saddle of a mountainbike."
  - "A sign that says 'Mountainbike Trail Ahead'."
  - Using the words "mountain" or "bike" in the prompts.
  - Mentioning parts like "handlebar," "gear shift," or "saddle."

Formatting Instructions:

- Start each prompt on a new line, numbered sequentially from 1 to 50.

- The format should be:

  1: <prompt_1>
  2: <prompt_2>
  3: <prompt_3>
  ...
  50: <prompt_50>

User Input Format:

The user will provide the object in the following format:

object: <object name>

Your Response:

- Return exactly 50 prompts per user request.

- Ensure that the last line of your response starts with:

  50: <prompt_50>

- Under no circumstances should you include any content in your response other than the 50 prompts. Do not include explanations, apologies, or any additional text.

Summary:

- Do not mention the object name or any part of it. If the object name is a composite or compound word, do not include any part of it in the prompts.

- Do not mention parts or components of the object.

- Do not create prompts that refer to written text containing the object name or any part of it.

- Focus on spurious features that frequently co-occur with the object.

- You may combine elements if they logically co-occur in an image.

- Ensure diversity and uniqueness in the prompts.

- Use proper language and avoid any inappropriate content.

- Review all prompts for compliance before submitting.

- Under no circumstances should you include any content in your response other than the 50 prompts. Do not include explanations, apologies, or any additional text.

Remember, the goal is to create prompts that could lead an AI model to falsely recognize the object due to the presence of spurious features, even though the object itself is not present in the images.

"""

FOLLOW_UP_PROMPT = """
Please review the list of prompts you previously generated and check for any mistakes or deviations from the guidelines. Identify any prompts that do not fully comply with the instructions. Then, generate a new list of 50 prompts that strictly adhere to all the guidelines provided.

Important Guidelines:

- Do not mention the object name or any part of it. If the object name is a composite or compound word, do not include any part of the object name in the prompts.
- Do not mention parts or components of the object.
- Do not create prompts that refer to written text containing the object name or any part of it.
- Focus on spurious features that frequently co-occur with the object.
- You may combine elements if they logically co-occur in an image.
- Ensure diversity and uniqueness in the prompts.
- Use proper language and avoid any inappropriate content.
- Review all prompts for compliance before submitting.
- Under no circumstances should you include any content in your response other than the 50 prompts. Do not include explanations, apologies, or any additional text.

Formatting Instructions:

- Start each prompt on a new line, numbered sequentially from 1 to 50.
- The format should be:

  1: <prompt_1>
  2: <prompt_2>
  3: <prompt_3>
  ...
  50: <prompt_50>

- Ensure that the last line of your response starts with:

  50: <prompt_50>

Remember, your goal is to create prompts that could lead an AI model to falsely recognize the object due to the presence of spurious features, even though the object itself is not present in the images.

Now, generate the corrected list of 50 prompts.
"""

@dataclass
class Args:
    source_file: str = 'data/spurious_imagenet.txt'
    result_dir: str = 'output_cvpr/1_prompts/spurious_imagenet'
    prompt_type: str = 'llama_only'
    num_beams: int = 1
    num_prompts: int = 50
    max_new_tokens: int = 10_000
    do_sample: bool = False
    batch_size: int = 1
    skip_existing: bool = False
    follow_up: bool = True

def setup() -> Args:
    default_config: Args = OmegaConf.structured(Args)
    cli_args = OmegaConf.from_cli()
    config: Args = OmegaConf.merge(default_config, cli_args)
    assert config.result_dir is not None
    return config

def main():
    args: Args = setup()


    object_list = []
    with open(args.source_file, 'r') as f:
        for line in f.readlines():
            object_list.append(line.strip())

    #
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

    #Load LLM
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=ACCESS_TOKEN,
    )

    #num prompts hardcoded into system prompt for better LLM understanding
    assert args.num_prompts == 50


    @torch.inference_mode()
    def make_prompts(objects: str | List[str]):
        if isinstance(objects, str):
            objects = [objects]

        prompts = []
        for obj in objects:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"object: {obj}"},
            ]
            prompts.append(messages)

        pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
        pipeline.tokenizer.padding_side = "left"

        outputs = pipeline(
            prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
        )

        object_to_initial_prompts = {}
        for obj, output in zip(objects, outputs):
            obj_response = output[0]["generated_text"][-1]['content']
            individual_prompts = obj_response.splitlines()
            object_to_initial_prompts[obj] = {}
            for idx, prompt in enumerate(individual_prompts):
                try:
                    idx, prompt = prompt.split(': ', 1)
                    object_to_initial_prompts[obj][idx] = prompt
                except:
                    print(f'broken prompt format: {prompt}')
                    continue

        if args.follow_up:
            prompts = []
            for obj, output in zip(objects, outputs):
                obj_response = output[0]["generated_text"][-1]['content']
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"object: {obj}"},
                    {"role": "assistant", "content": obj_response},
                    {"role": "user", "content": FOLLOW_UP_PROMPT},
                ]
                prompts.append(messages)

            pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id[0]
            pipeline.tokenizer.padding_side = "left"

            outputs = pipeline(
                prompts,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
            )

            object_to_follow_up_prompts = {}
            for obj, output in zip(objects, outputs):
                obj_response = output[0]["generated_text"][-1]['content']
                individual_prompts = obj_response.splitlines()
                object_to_follow_up_prompts[obj] = {}
                for idx, prompt in enumerate(individual_prompts):
                    try:
                        idx, prompt = prompt.split(': ', 1)
                        object_to_follow_up_prompts[obj][idx] = prompt
                    except:
                        print(f'broken prompt format: {prompt}')
                        continue
        else:
            object_to_follow_up_prompts = None

        return object_to_initial_prompts, object_to_follow_up_prompts

    while object_list:
        batch = []
        while len(batch) < args.batch_size and object_list:
            obj = object_list.pop()
            prompts_json = os.path.join(args.result_dir, f"{obj}_prompts_{args.prompt_type}_{args.num_prompts}.json")
            if not (args.skip_existing and os.path.isfile(prompts_json)):
                batch.append(obj)

        batch_prompts_initial, batch_prompts_follow_up = make_prompts(batch)
        for obj in batch:
            prompts_json = os.path.join(args.result_dir, f"{obj}_prompts_{args.prompt_type}_{args.num_prompts}.json")
            with open(prompts_json, 'w', encoding='utf-8') as f:
                object_prompts = batch_prompts_initial[obj]
                json.dump(object_prompts, f, ensure_ascii=False, indent=4)

            if batch_prompts_follow_up is not None:
                prompts_json = os.path.join(args.result_dir,
                                            f"{obj}_prompts_{args.prompt_type}_follow_up_{args.num_prompts}.json")
                with open(prompts_json, 'w', encoding='utf-8') as f:
                    object_prompts = batch_prompts_follow_up[obj]
                    json.dump(object_prompts, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
