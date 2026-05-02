# Colab Run Steps

## 1. Install packages
```python
!pip install -r requirements.txt
```

## 2. Train the model
Quick test:
```python
!python train_model.py --sample-size 30000 --test-size 10000
```

Final version:
```python
!python train_model.py --sample-size 150000 --test-size 30000
```

## 3. Start Streamlit
```python
!pkill -f streamlit
!nohup streamlit run app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --server.enableWebsocketCompression false \
  > streamlit.log 2>&1 &
```

## 4. Start Cloudflare Tunnel
```python
!pkill -f cloudflared
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501
```

Open the `trycloudflare.com` link.
