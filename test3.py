from eval_utils import evaluate_model

accuracy_1 = evaluate_model("/teamspace/studios/this_studio/dataset_gen/finetuned_models/Qwen2.5-3B_3dae03e5-ce03-4a11-b8c7-b4e144b3d80c","/teamspace/studios/this_studio/dataset_gen/generated_datasets/ac80fff99668/test.json")

print(f"accuracy of the finetuned model is:{accuracy_1}")

accuracy_2 = evaluate_model("/teamspace/studios/this_studio/qwen2.5-3b","/teamspace/studios/this_studio/dataset_gen/generated_datasets/ac80fff99668/test.json") 

print(f"accuracy of the unquantized model is:{accuracy_2}")