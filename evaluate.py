from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Sample data
data = {
    "question": ["What is KYC?", "How to report fraud?"],
    "answer": [
        "KYC is identity verification process for banking customers.",
        "Report fraud to bank helpline."
    ],
    "contexts": [
        ["KYC is mandatory for all banking customers..."],
        ["Report fraud immediately to the bank..."]
    ],
    "ground_truth": [
        "KYC includes identity and address verification.",
        "Fraud must be reported immediately."
    ]
}

dataset = Dataset.from_dict(data)

# Evaluate
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(result)