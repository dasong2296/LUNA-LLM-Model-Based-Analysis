import openai
import os
import numpy as np
import joblib

def evaluate_summary(instance):
    prompt = (
        f"Input code: {instance['input']}\n"
        f"Ground Truth Summary: {instance['ground_truth']}\n"
        f"Generated Summary: {instance['output']}\n"
        "Evaluate the quality of the Generated Summary compared to the Ground Truth Summary. "
        "If the Generated Summary is accurate and relevant, respond with 1. If it is poor or irrelevant, respond with 0."
    )

    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",  # or "gpt-4" depending on the exact engine name
        prompt=prompt,
        max_tokens=10
    )

    # Interpret response as binary label
    return 1 if response.choices[0].text.strip() == "1" else 0


file_name = "../../../../../../data/songda/tl_code_sum/31/codellama-13b-Instruct.joblib"
file_with_scores = "../../../../../../data/songda/tl_code_sum/31/codellama-13b-Instruct_with_semantic.joblib"

fp = open(file_name, "rb")
fw = open(file_with_scores, "ab+")

embed_set = []
data_points = []

i = 0 
while i <= 5000:
    try:
        instance = joblib.load(fp)
        
        quality_label = evaluate_summary(instance)
        print(f"Quality of Summary (1: Good, 0: Bad): {quality_label}")
        print("============") 
        instance["binary_label"] = quality_label
        
        joblib.dump(instance, fw)
        print("dump {}th record".format(i))
        i +=1
        
    except Exception as e:
        print(e)
        break
print("finish")