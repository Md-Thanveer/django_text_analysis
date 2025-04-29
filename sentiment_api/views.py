import torch
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import SentimentModel
# your model class
from .tokenizer import simple_tokenizer  # your tokenizer

# Load model
model = SentimentModel(vocab_size=5000, embed_dim=128, hidden_dim=256, output_dim=2)
model.load_state_dict(torch.load('sentiment_model.pth', map_location=torch.device('cpu')))
model.eval()


class SentimentAnalysisView(APIView):
    def post(self, request, *args, **kwargs):
        text = request.data.get('text')

        if not text:
            return Response({"error": "No text provided"}, status=status.HTTP_400_BAD_REQUEST)

        tokens = simple_tokenizer(text)  # you write this function
        tokens_tensor = torch.tensor(tokens).unsqueeze(0)  # batch_size=1

        with torch.no_grad():
            output = model(tokens_tensor)
            prediction = torch.argmax(output, dim=1).item()

        sentiment = "Positive" if prediction == 1 else "Negative"

        return Response({"sentiment": sentiment})