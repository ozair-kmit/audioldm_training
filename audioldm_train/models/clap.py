class CLAPTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Simplified CLAP text encoder
        from transformers import AutoTokenizer, AutoModel
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        self.projection = nn.Linear(config.hidden_size, config.embed_dim)
        
    def forward(self, texts):
        # Tokenize texts
        inputs = self.tokenizer(
            texts, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=77
        )
        
        # Get text embeddings
        outputs = self.text_encoder(**inputs)
        text_embed = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length
        
        # Project to desired dimension
        return self.projection(text_embed)
