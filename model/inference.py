import torch, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataclasses import dataclass
from data.load_datasets import ID2LABEL

@dataclass
class ClassificationResult:
    label:       str
    label_id:    int
    confidence:  float
    is_threat:   bool
    scores:      dict[str, float]
    latency_ms:  float
    explanation: str


class AdversarialShieldClassifier:
    """
    Singleton-ready inference class.
    Load once at startup; call .classify() per request.
    """
    _SAFE_LABEL = "SAFE"

    def __init__(self, model_path: str, threshold: float = 0.75):
        self.threshold = threshold
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[AdversarialShield] Loading model from {model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("[AdversarialShield] ✅ Ready")

    @torch.no_grad()
    def classify(self, text: str) -> ClassificationResult:
        t0 = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        outputs    = self.model(**inputs)
        probs      = torch.softmax(outputs.logits, dim=-1)[0]
        label_id   = probs.argmax().item()
        confidence = probs[label_id].item()
        label      = ID2LABEL[label_id]
        latency_ms = (time.perf_counter() - t0) * 1000

        scores = {ID2LABEL[i]: round(p.item(), 4) for i, p in enumerate(probs)}
        is_threat = (label != self._SAFE_LABEL) and (confidence >= self.threshold)

        return ClassificationResult(
            label=label,
            label_id=label_id,
            confidence=round(confidence, 4),
            is_threat=is_threat,
            scores=scores,
            latency_ms=round(latency_ms, 2),
            explanation=self._explain(label, confidence, text),
        )

    def _explain(self, label: str, conf: float, text: str) -> str:
        """Generate a human-readable explanation for the classification."""
        explanations = {
            "SAFE":               "Prompt appears benign. No adversarial patterns detected.",
            "JAILBREAK":          "Role-override or persona manipulation detected (e.g. DAN, 'pretend you are').",
            "PROMPT_INJECTION":   "Injection attempt found — likely 'ignore previous instructions' pattern.",
            "HARMFUL_CONTENT":    "Request contains or solicits harmful, dangerous, or illegal content.",
            "DATA_EXFILTRATION":  "Prompt attempts to extract system instructions or internal context.",
            "SOCIAL_ENGINEERING": "Authority impersonation or urgency manipulation pattern detected.",
        }
        base = explanations.get(label, "Unknown pattern.")
        return f"{base} Confidence: {conf:.1%}."

    def batch_classify(self, texts: list[str]) -> list[ClassificationResult]:
        return [self.classify(t) for t in texts]