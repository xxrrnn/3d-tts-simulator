"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

COT_EXAMPLES = """
<|im_start|>user
A $90^\\circ$ rotation around the origin in the counter-clockwise direction is applied to $7 + 2i.$  What is the resulting complex number?<|im_end|>
<|im_start|>assistant
Step 1: To rotate a complex number by $90^\\circ$ counter-clockwise around the origin, we can multiply the complex number by $i$. This is because multiplication by $i$ corresponds to a $90^\\circ$ counter-clockwise rotation in the complex plane

Step 2: The given complex number is $7 + 2i$. Let's perform the multiplication:\n\\[ (7 + 2i) \\cdot i \\]

Step 3: Apply the distributive property of multiplication over addition:\n\\[ 7i + 2i^2 \\]

Step 4: Recall that $i^2 = -1$. Substitute this into the expression:\n\\[ 7i + 2(-1) \\]

Step 5: Simplify the expression:\n\\[ 7i - 2 \\]

Step 6: Therefore, the final answer is:\n\\(\\boxed{-2 + 7i}\\)\n
"""

COT_TASK_DESC = "Please reason step by step, and put your final answer within \\boxed{}."

PROBLEM_FORMAT_STR = "{question}"

SEP = "\n\n"
