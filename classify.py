from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

text = "I love DistilBERT"
result = classifier(text)

print("Text:", text)
print("Prediction:", result)