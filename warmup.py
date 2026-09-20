from src.chunking import FixedSizeChunker

text = "a" * 10000
for ov in (50, 100):
    chunks = FixedSizeChunker(chunk_size=500, overlap=ov).chunk(text)
    print(f"overlap={ov}: {len(chunks)} chunk")