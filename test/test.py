from pymilvus import MilvusClient
client = MilvusClient(uri="http://127.0.0.1:19530")
result = client.query(
    collection_name="paper_item_names_v2",
    filter='paper_title == "Backpropagation-Free Test-Time Adaptation via Probabilistic Gaussian Alignment"',
    output_fields=["paper_title", "file_title"]
)
print(result)
