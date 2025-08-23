from src.pipeline.text.labels import Labels

label = Labels(dynamic = True ,path ="facebook/bart-large-mnli")
texts =[
    "I love playing football on weekends.",
    "The government passed a new law today.",
    "The new iPhone has amazing features."
]
cad_labels =['sports','tech',"politics"]

output = label(texts,labels =cad_labels,multilabel =False)

for i , result in enumerate(output):
    print(f"\n Text : {texts[i]}")

    for idx,score in result:
        print(f"{cad_labels[idx]} :{ score:.4f}")


print("-----------------------Text-Classification------------------------------------------")
labels = Labels(dynamic = False, path ="distilbert-base-uncased-finetuned-sst-2-english")
texts =[
    "I really like this movie!",
    "This was the worst film I’ve ever seen.",
    "The acting was decent but the story was boring."
]
outputs =labels(texts,labels=['positive','negative'])

for i , result in enumerate(outputs):

    print(f"\n Text : {texts[i]}")

    for idx,score in result:
        print(f"{labels.pipeline.model.config.id2label[idx]}: {score:.4f}")
