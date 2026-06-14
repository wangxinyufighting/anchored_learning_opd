from vllm import LLM, SamplingParams
from typing import List, Dict
import json
import os
import re
import torch

# from  examples.data_preprocess.gsm8k import instruction_following as gsm8k_prompt

mmlu_pro_prompt = """
Reason step by step about the correct answer based on the question and options provided. 
After your reasoning, you will select the most correct answer(e.g., A, B, C, D, F, G, H, I, J) and write it in \\boxed{}. 
For example: \\boxed{A}
Let's think step by step!
"""

med_qa_prompt = """
Reason step by step about the correct answer based on the question and options provided. 
After your reasoning, you will select the most correct answer(e.g., A, B, C, D) and write it in \\boxed{}. For example: \\boxed{A}

The complete format for answering the question is as follows:
<\\analysis> ... <\\analysis> 
<\\answer> \\boxed{}

Please answer strictly in accordance with the above format.
"""

iGSM_prompt = """
Let\'s think step by step and output the final answer in \\boxed{}.
"""
# iGSM_prompt = """
# Solve the following math problem step by step. Provide your final answer in the format \\boxed{answer}.\n\nProblem: 
# """

TCM_prompt = """
你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D，E中选出一个最正确的答案，并写在\\boxed{}中。
例如：\\boxed{A}

完整的题目回答的格式如下：
【解析】 ... <eoe>
 \\boxed{}
请你严格按照上述格式作答。
"""

finance_benchmark_mcq_prompt = """
Please think step by step and write your thinking process between 【Analysis】 and <eoe>. 
You will select the most correct answer from A, B, C, D, and E and write it in \\boxed{}.
For example: \\boxed{A}
The complete format for answering the question is as follows:【Analysis】 ... <eoe> \\boxed{}
"""

# def extract_TCM_solution(solution_str):
#     solution = re.search("(?<=boxed{)[a-zA-Z]", solution_str)

#     if solution is not None:
#         final_solution = solution.group(0)
#     else:
#         if 'answer is:\n\n' in solution_str:
#             solution = re.search("(?<=answer is:\\n\\n)[a-zA-Z]", solution_str)
#             if solution is not None:
#                 final_solution = solution.group(0)
#         final_solution = ""

#     return final_solution

def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False
    
def evaluate_equation(equation_str):
    try:
        """Safely evaluate the arithmetic equation using eval() with precautions."""
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'^[\d+\-*/().\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except:
        return -100000000

def is_right_countdown(prediction, gt_answer, nums):
    pred = extract_countdonw_solution(prediction)

    if pred:
        pred = pred.strip()
        if validate_equation(pred, nums):
            pred_value = evaluate_equation(pred)
            if pred_value == -100000000:
                print("计算错误")
            return float(gt_answer) == float(pred_value), pred, gt_answer
                
    return False, pred, gt_answer

def is_right_medcalc(prediction, gt_answer, upper=None, lower=None):
    pred = extract_medcalc_solution(prediction, upper)

    answer = None
    
    if pred:
        pred = pred.strip()
        
        if 'weeks' in upper:
            gt_answer = gt_answer.replace('\'', '')
            gt_answer = gt_answer.replace("'", "")
            gt_answer = gt_answer.replace(" ", "")
            
            pred = pred.replace('\'', '')
            pred = pred.replace("'", "")
            pred = pred.replace(" ", "")
        
        if str(gt_answer) == str(pred):
            return True, pred, gt_answer
        
        if upper == lower:
            answer = (gt_answer == pred)
        else:
            try:
                answer = (float(upper) >= float(pred) and float(lower) <= float(pred))
            except:
                answer = False
            
        return answer, pred, gt_answer
                
    return False, pred, gt_answer

def extract_medcalc_solution(solution_str, upper_bound):
    # extrect answer from <answer>{\"answer\": \"('15 weeks', '4 days')\"}</answer>\n
    solution_week_and_day = re.search("(?<=<answer>{\"answer\": \").*(?=\"}</answer>)", solution_str)
    
    if "weeks" in upper_bound and "days" in upper_bound:
        if solution_week_and_day is not None:
            final_solution = solution_week_and_day.group(0)
        else:
            final_solution = "-1000000000"
        return final_solution
    
    # extract answer from format <answer>{\"answer\": str(short and direct answer of the question)}</answer>\
    solution = re.search("(?<=<answer>{\"answer\": \").*(?=\"}</answer>)", solution_str)
    solution2 = re.search("(?<=<answer>{\"answer\": \").*(?=\"}}</answer>)", solution_str)
    # 获取最后一个出现的整数或小数
    # solution3 = re.search("-?\d+(?:[.,]\d+)*", solution_str[::-1])

    if solution is not None:
        final_solution = solution.group(0)
        if ' ' in final_solution:
            final_solution = final_solution.split(' ')[0]
    elif solution2 is not None:
        final_solution = solution2.group(0)
        if ' ' in final_solution:
            final_solution = final_solution.split(' ')[0]
    # elif solution3 is not None:
        # final_solution = solution3.group(0)[::-1]
    else:
        final_solution = "-1000000000"

    return final_solution

def extract_countdonw_solution(solution_str):
    solution = re.search("(?<=boxed{).*(?=\})", solution_str)

    if solution is not None:
        final_solution = solution.group(0)
        if '\\times' in final_solution:
            final_solution = final_solution.replace('\\times', '*')
    else:
        final_solution = None

    return final_solution

def extract_iGSM_solution(solution_str):
    solution = re.search("(?<=boxed{)-?\d+(?:[.,]\d+)*(?=\})", solution_str)

    if solution is not None:
        final_solution = solution.group(0)
    else:
        final_solution = "-1000000000"

    return final_solution

def extract_TCM_solution(solution_str, prompt=""):
    if '\boxed' in solution_str:
        solution_str = solution_str.replace('\\boxed', 'boxed')
    if 'Boxed' in solution_str:
        solution_str = solution_str.replace('Boxed', 'boxed')

    solution = re.search("(?<=boxed{)[a-zA-Z]", solution_str)

    final_solution = ""
    if solution is not None:
        final_solution = solution.group(0)
    else:
        if 'answer is:\n\n' in solution_str:
            # print(solution_str)
            solution = re.search("(?<=answer is:\n\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is: ' in solution_str:
            solution = re.search("(?<=answer is: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is ' in solution_str:
            solution = re.search("(?<=answer is )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Answer: ' in solution_str:
            solution = re.search("(?<=Answer\: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer: ' in solution_str:
            solution = re.search("(?<=answer: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Answer\n' in solution_str:
            solution = re.search("(?<=Final Answer\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'answer is:\n' in solution_str:
            solution = re.search("(?<=answer is:\n)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'Final Answer**: ' in solution_str:
            solution = re.search("(?<=Final Answer\*\*: )\w", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif solution_str in prompt:
            final_solution = solution_str
        elif 'which is option ' in solution_str:
            solution = re.search("(?<=which is option )[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'boxed: ' in solution_str:
            solution = re.search("(?<=boxed: )[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        elif 'is **' in solution_str:
            solution = re.search("(?<=is \*\*)[a-zA-Z]", solution_str)
            if solution is not None:
                final_solution = solution.group(0)
        else:
            solution = re.search("^[a-zA-Z]\.", solution_str)
            if solution is not None:
                final_solution = solution.group(0)[0]
            else:
                final_solution = ""

    return final_solution

def extract_mmlu_pro_solution(solution_str):
    solution = re.search("#### [a-zA-Z](?=[^a-zA-Z])", solution_str)

    if solution is not None:
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
    else:
        if '####' in solution_str:
            solution_str = solution_str.split('####')[-1]
        solution = re.findall("[a-zA-Z](?=[^a-zA-Z])", solution_str)
        if solution is not None and len(solution) > 0:
            final_solution = solution[-1]
        else:
            final_solution = ""

    while final_solution and len(final_solution) > 1:
        final_solution = final_solution[:-1]

    return final_solution

def is_right_mmlu_pro(prediction:str, gt:str) -> float:
    pred = extract_mmlu_pro_solution(prediction)

    if pred and gt:
        gt = gt.strip()
        pred = pred.strip()
        while pred[-1] == '.':
            pred = pred[:-1]

        return gt.lower() == pred.lower(), pred, gt
        
    return False, pred, gt

def is_right_TCM(prediction:str, gt:str) -> float:
    pred = extract_TCM_solution(prediction)

    if pred and gt:
        gt = gt.strip()
        pred = pred.strip()
        while pred[-1] == '.':
            pred = pred[:-1]

        return gt.lower() == pred.lower(), pred, gt
        
    return False, pred, gt


def extract_gsm8k_solution(solution_str):
    solution_str = solution_str.strip()
    solution_str = solution_str[:-1] if len(solution_str) > 0 and solution_str[-1] == '.' else solution_str 
    solution = re.search("#### -?\d+(?:[.,]\d+)*", solution_str)
    if solution is not None:
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
    else:
        solution = None
        final_solution = "-1000000000"
        if 'boxed' in solution_str:
            solution = re.search("(?<=boxed{)-?\d+(?:[.,]\d+)*", solution_str)
        elif '####' in solution_str:
            solution_str = solution_str.split('####')[-1]
            if 'frac' not in solution_str:
                solution = re.search("-?\d+(?:[.,]\d+)*", solution_str)
        elif '###' in solution_str:
            solution_str = solution_str.split('###')[-1]
            if 'frac' not in solution_str:
                solution = re.search("-?\d+(?:[.,]\d+)*", solution_str)

        if solution is not None:
            final_solution = solution.group(0)
            final_solution = final_solution.replace(",", "")
        else:
            solutions = re.findall(r"-?[0-9.,]+", solution_str)
            if len(solutions) > 0:
                for i in solutions[::-1]:
                    if i != '.':
                        final_solution = i
                        break
            else:
                final_solution = "-1000000000"

    if final_solution and final_solution[-1] == '.':
        final_solution = final_solution[:-1]
    return final_solution


def is_right_gsm8k(prediction:str, ground_truth:str) -> float:
    gt = extract_gsm8k_solution(ground_truth)
    pred = extract_gsm8k_solution(prediction)

    if pred and gt:
        gt = gt.replace(',', '')
        pred = pred.replace(',', '').strip()
        if len(pred) > 0 and pred != '.':
            while pred[-1] == '.':
                pred = pred[:-1]
                if pred == '':
                    break
        try:
            return float(gt) == float(pred), pred, gt
        except:
            return False, pred, gt
        
    return False, pred, gt

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class DirectBatchInference:
    def __init__(self, model_path: str):
        print("正在加载vLLM模型...")
        # 自动检测或使用指定的 GPU 数量
        num_gpus = None
        if num_gpus is None:
            # 从环境变量读取
            cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if cuda_devices:
                num_gpus = len(cuda_devices.split(","))
            else:
                # 使用 torch 检测
                num_gpus = torch.cuda.device_count()

        # 自动检测模型系列
        # if model_family == "auto":
        model_path_lower = model_path.lower()
        if "llama" in model_path_lower or "llama" in model_path_lower:
            model_family = "llama"
        elif "qwen" in model_path_lower:
            model_family = "qwen"
        else:
            model_family = "llama"  # 默认使用 llama 的停止词

        # 设置不同模型的停止词
        self.stop_tokens = self._get_stop_tokens(model_family)

        print(f"🚀 使用 {num_gpus} 个 GPU 进行推理")
        print(f"📝 检测到模型系列: {model_family}")
        if num_gpus > 1:
            print(f"   模式: 张量并行 (tensor_parallel_size={num_gpus})")
        print(f"🛑 停止词: {self.stop_tokens}")

        self.model = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=4096,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=0.9,
            enforce_eager=True,
            max_num_seqs=4096,
            enable_prefix_caching=True,
            dtype=torch.bfloat16,
        )
        print("✅ vLLM模型加载完成!")
        print(f"   可用GPU: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    def _get_stop_tokens(self, model_family: str) -> list:
        """根据模型系列返回相应的停止词"""
        if model_family == "llama":
            # LLaMA 3/3.1/3.2 的停止词
            return ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]
        elif model_family == "qwen":
            # Qwen2/2.5 的停止词
            return ["<|endoftext|>", "<|im_end|>"]
        else:
            # 默认停止词
            return ["<|eot_id|>", "<|end_of_text|>", "<|im_end|>"]


    def batch_inference(self, batch_messages,
                       max_tokens: int = 512, temperature: float = 0.7) -> List[str]:

        print(f"准备了 {len(batch_messages)} 条消息")

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            stop=self.stop_tokens,  # 使用模型特定的停止词
            n=1  # 生成多个序列
        )

        outputs = self.model.chat(
                messages=batch_messages,
                sampling_params=sampling_params,
                use_tqdm=True  # 显示进度
            )

        responses = []
        for output in outputs:
            # print(output)
            if output and output.outputs:
                responses.append(output.outputs[0].text)
            else:
                responses.append("生成失败")

        return responses

def load_datasets(test_file: str) -> List[Dict]:
    with open(test_file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def prepare_messages(datasets, prompt=None) -> List[Dict]:
    batch_messages = []
    for i in datasets:
        new_data = i['messages'][0]
        if prompt is not None:
            question = i['messages'][0]['content']
            new_data['content'] = f'{question} {prompt}'
        else:
            new_data = i['messages'][0]
        batch_messages.append([new_data])

    return batch_messages


def main_direct_batch(args):
    inferencer = DirectBatchInference(args.model_path)
    datasets = load_datasets(args.test_file)
    data_name = None
    if 'gsm8k' in args.test_file:
        # batch_messages = prepare_messages(datasets, gsm8k_prompt)
        batch_messages = prepare_messages(datasets, iGSM_prompt)
        data_name = 'gsm8k'
    elif 'mmlu_pro' in args.test_file:
        batch_messages = prepare_messages(datasets, mmlu_pro_prompt)
        data_name = 'mmlu_pro'
    elif 'TCM' in args.test_file:
        batch_messages = prepare_messages(datasets, TCM_prompt)
        data_name = 'TCM'
    elif 'finance_benchmark_mcq' in args.test_file:
        batch_messages = prepare_messages(datasets, finance_benchmark_mcq_prompt)
        data_name = 'finance_benchmark_mcq'
    elif 'med_qa' in args.test_file:
        batch_messages = prepare_messages(datasets, med_qa_prompt)
        data_name = 'med_qa'
    elif 'iGSM' in args.test_file:
        batch_messages = prepare_messages(datasets, iGSM_prompt)
        data_name = 'iGSM'
    elif 'countdown' in args.test_file:
        batch_messages = prepare_messages(datasets)
        data_name = 'countdown'
    elif 'medcalc' in args.test_file:
        batch_messages = prepare_messages(datasets)
        data_name = 'medcalc'

    print(batch_messages[0])
    print(f"加载了 {len(batch_messages)} 条检测数据")

    import datetime
    # ts = int(time.time())

    responses = inferencer.batch_inference(
        batch_messages=batch_messages,
        temperature=args.temperature,
        max_tokens = args.max_tokens
    )

    all_correct = 0
    parent_path = f"./results/responses/{data_name}"
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    output_file = f"{parent_path}/results_{datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')}.json"

    empty_result_num = 0

    with open(output_file, 'w') as w:
        for data, response in zip(datasets, responses):
            extra_info = {}
            gt = data['messages'][1]['content']
            if 'gsm8k' in args.test_file or 'iGSM' in args.test_file:
                is_right, pred_answer, gt_answer = is_right_gsm8k(response, gt)
            # elif 'mmlu_pro' in args.test_file or 'TCM' in args.test_file:
            #     is_right, pred_answer, gt_answer = is_right_mmlu_pro(response, gt)
            elif 'mmlu_pro' in args.test_file or 'TCM' in args.test_file or 'finance_benchmark_mcq' in args.test_file or 'med_qa' in args.test_file:
                is_right, pred_answer, gt_answer = is_right_TCM(response, gt)
            elif 'countdown' in args.test_file:
                gt = data['y']
                nums = data['x']
                extra_info['nums'] = nums
                is_right, pred_answer, gt_answer = is_right_countdown(response, gt, nums)
            elif 'medcalc' in args.test_file:
                upper_bound = data['upper_bound']
                lower_bound = data['lower_bound']
                extra_info['upper_bound'] = upper_bound
                extra_info['lower_bound'] = lower_bound
                is_right, pred_answer, gt_answer = is_right_medcalc(response, gt, upper=upper_bound, lower=lower_bound)

            if pred_answer == "" or pred_answer == "-1000000000":
                empty_result_num += 1

            all_correct += 1 if is_right else 0
                
            result = {
                'prompt': data['messages'][0]['content'],
                'is_right': is_right,
                'ground_truth': gt,
                'extracted_gt': gt_answer,
                'prediction': response,
                'extracted_pred': pred_answer,
                'extra_info': extra_info
            }
            w.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # for data, response in zip(datasets, responses):
    #     gt = data['messages'][1]['content']
    #     if 'gsm8k' in args.test_file or 'iGSM' in args.test_file:
    #         is_right, pred_answer, gt_answer = is_right_gsm8k(response, gt)
    #     # elif 'mmlu_pro' in args.test_file or 'TCM' in args.test_file:
    #     #     is_right, pred_answer, gt_answer = is_right_mmlu_pro(response, gt)
    #     elif 'mmlu_pro' in args.test_file or 'TCM' in args.test_file or 'finance_benchmark_mcq' in args.test_file or 'med_qa' in args.test_file:
    #         is_right, pred_answer, gt_answer = is_right_TCM(response, gt)
    #     elif 'countdown' in args.test_file:
    #         gt = data['y']
    #         nums = data['x']
    #         is_right, pred_answer = is_right_countdown(response, gt, nums)
    #     elif 'medcalc' in args.test_file:
    #         is_right, pred_answer = is_right_medcalc(response, gt)

    #     all_correct += 1 if is_right else 0
        

    print(f"结果已保存到: {output_file}")
    empty_rate = empty_result_num/len(datasets)
    
    acc = all_correct/len(datasets)
    with open(args.output_dir, 'a') as f:
        # f.write(f"{args.test_file}\t{acc:.4f}\t{empty_rate:.4f}\t{args.max_tokens}\t{args.temperature}\t{args.model_path}\n")
        f.write(f"{args.model_path}\t{acc:.4f}\t{empty_rate:.4f}\t{args.test_file}\t{args.max_tokens}\t{args.temperature}\n")

    print(args.model_path)
    print(args.test_file)
    print(f"准确率: {all_correct}/{len(datasets)} = {acc:.4f}")
    print(f"空结果数: {empty_result_num} / {len(datasets)} = {empty_rate:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,  default='/gemini/space/models/qwen2.5-vl-72b-instruct')
    parser.add_argument('--test_file', type=str, default = '/gemini/space/guanming/sft/datasets/iGSM/conversational_language_modeling/test_16.json')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--output_dir', type=str, default='./results/results.6.0.txt')

    args = parser.parse_args()

    # 推荐使用直接批量推理方案
    main_direct_batch(args)
