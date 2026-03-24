import argparse
from pathlib import Path

from transformers import AutoTokenizer

# 复用 compute-optimal-tts 里的 RM 调用逻辑
from reason.inference.rm_call import (
    RMRemoteCaller,
    RemoteRewardModelConfig,
    get_prm_special_tokens,
)

ROOT = Path(__file__).resolve().parents[2]  # .../3d-tts-simulator
DEFAULT_RM_PATH = "/DISK1/data/rnxu_24/Paper/3d-tts-simulator/data/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B"


def build_rm_caller(rm_model_path: str, controller_addr: str) -> RMRemoteCaller:
    """
    构造一个 Skywork PRM 的 RMRemoteCaller（走 fastchat worker）。
    """
    rm_model_path = str(Path(rm_model_path).expanduser())
    rm_model_name = rm_model_path

    tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
    step_tag_id, returned_token_ids = get_prm_special_tokens(rm_model_name, tokenizer)

    # Skywork 默认使用 "{question} {answer}" 格式
    prm_format_str = "{question} {answer}"

    rm_config = RemoteRewardModelConfig(
        prm_step_tag="\n",               # 对 Skywork，真正的 step 分隔是在 infer_fns 里处理，这里随便给个占位
        format_str=prm_format_str,
        model_name=rm_model_name,
        controller_addr=controller_addr,
        step_tag_id=step_tag_id,
        returned_token_ids=returned_token_ids,
        rm_serve_type="fastchat",
        multi_gpu=False,
    )
    rm_call = RMRemoteCaller(rm_config, tokenizer=tokenizer)
    return rm_call


def main():
    parser = argparse.ArgumentParser(
        description="使用 Skywork PRM 对自定义 (question, answer) 做 reward 评估（通过 fastchat worker）"
    )
    parser.add_argument(
        "--rm-path",
        type=str,
        default=str(DEFAULT_RM_PATH),
        help="Skywork PRM 模型路径，本地目录，例如 data/models/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",
    )
    parser.add_argument(
        "--controller-addr",
        type=str,
        default="http://localhost:10014",
        help="FastChat controller 地址（需要先用 scripts/eval.sh / serve_gpu4.sh 启动 reward worker）",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="A digital display shows the current date as an $8$-digit integer consisting of a $4$-digit year, followed by a $2$-digit month, followed by a $2$-digit date within the month. For example, Arbor Day this year is displayed as 20230428. For how many dates in $2023$ will each digit appear an even number of times in the 8-digital display for that date?",
        help="要评估的题目（question）",
    )
    parser.add_argument(
        "--answer",
        type=str,
        required=False,
        default="To determine how many dates in 2023 will have each digit appearing an even number of times in the 8-digit display, we need to analyze the structure of the date format, which is YMMDD (_year, month, day).ки\\n To determine how many dates in 2023 will have each digit appearing an even number of times in their 8-digit display, we need to analyze the structure of the date format: YMMDD, where Y is the year, M is the month, and D is the day.",
        help="模型生成的分步推理答案（answer），内部用 ' ки\\n' 分隔 step",
    )
    args = parser.parse_args()

    print(f"[INFO] 使用 Skywork PRM 模型: {args.rm_path}")
    print(f"[INFO] FastChat controller: {args.controller_addr}")

    rm_call = build_rm_caller(args.rm_path, args.controller_addr)

    # RMRemoteCaller 期望的输入：List[(question, answer)]
    qa_pairs = [(args.question, args.answer)]
    model_names = ["dummy_lm_name"]  # 这里只是占位，内部不会真正用到

    # 通过 fastchat worker 调用 Skywork PRM 评估 reward
    rewards = rm_call(qa_pairs, model_names=model_names, verbose=False, local=False)
    print("===== Skywork PRM Reward 结果 =====")
    print("question:", args.question)
    print("answer:\n", args.answer)
    print("rewards:", rewards)


if __name__ == "__main__":
    main()