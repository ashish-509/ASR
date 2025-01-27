import time

start = time.time()
segments, _ = model.transcribe(np.random.randn(16000*2), language="ne")
print(f"Latency: {time.time()-start:.2f}s")  # Should be <0.5s