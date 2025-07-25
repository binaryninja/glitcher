# Hugging Face Authentication for Glitcher

This guide helps you set up Hugging Face authentication to access gated models like Llama, Mistral, and other state-of-the-art language models.

## Why Authentication is Needed

Many cutting-edge models are "gated" - they require:
- ✅ Hugging Face account
- ✅ Valid API token  
- ✅ Explicit access approval from model creators

**Gated Models Include:**
- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct` 
- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

**Public Models (No Auth):**
- `microsoft/DialoGPT-medium`
- `gpt2`, `distilbert-base-uncased`

## Quick Setup (3 Steps)

### 1. Get Your Token
1. **Sign up**: [huggingface.co](https://huggingface.co)
2. **Generate token**: [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Copy token**: Format: `hf_xxxxxxxxxxxxxxxxxxxxxxxxx`

### 2. Request Model Access
Visit each model page and click "Request access":
- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [Llama-3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

⏰ **Wait time**: Minutes to hours for approval

### 3. Configure Authentication

**Option A: Makefile (Recommended)**
```bash
# Setup with your token (saves to .env)
make hf-setup TOKEN=hf_your_token_here

# Check status
make hf-status
```

**Option B: Environment Variable**
```bash
# Set for current session
export HF_TOKEN=hf_your_token_here

# Add to .env file permanently
echo "HF_TOKEN=hf_your_token_here" >> .env
```

**Option C: Interactive Setup**
```bash
# Interactive login (saves to .env)
make hf-login
```

## Testing Authentication

### Check Status
```bash
# Comprehensive auth check
make hf-status
```

### Test with Demo
```bash
# With authentication (uses Llama)
export HF_TOKEN=hf_your_token_here
make demo

# Without authentication (uses public model)
make demo
```

### Test Mining
```bash
# Mine with Llama model
HF_TOKEN=hf_your_token_here make mine

# Mine with custom model
HF_TOKEN=hf_your_token_here make mine MODEL=meta-llama/Llama-3.2-3B-Instruct
```

## Docker Usage Examples

### Direct Docker Commands
```bash
# Run with authentication
docker run --rm --gpus all \
  -v ./models:/app/models \
  -v ./outputs:/app/outputs \
  -e HF_TOKEN=hf_your_token_here \
  glitcher:gpu \
  glitcher mine meta-llama/Llama-3.2-1B-Instruct

# Development container with auth
docker run -it --gpus all \
  -v $(PWD):/app \
  -e HF_TOKEN=hf_your_token_here \
  glitcher:dev bash
```

### Docker Compose
```bash
# Set in environment
export HF_TOKEN=hf_your_token_here

# Or add to .env file
echo "HF_TOKEN=hf_your_token_here" >> .env

# Start services
docker-compose up glitcher-gpu
```

## Verification Checklist

✅ **Token Valid**
```bash
make hf-status
# Should show: ✅ Token is valid for user: your_username
```

✅ **Model Access**
```bash
make hf-status  
# Should show: ✅ meta-llama/Llama-3.2-1B-Instruct - Accessible
```

✅ **Mining Works**
```bash
HF_TOKEN=your_token make mine MODEL=meta-llama/Llama-3.2-1B-Instruct
# Should start downloading and mining
```

## Troubleshooting

### ❌ "No Hugging Face token found"
**Solution**: Set HF_TOKEN environment variable
```bash
export HF_TOKEN=hf_your_token_here
# OR
make hf-setup TOKEN=hf_your_token_here
```

### ❌ "401 Client Error: Unauthorized"
**Causes**: 
- Invalid token
- Token expired
- No access to model

**Solutions**:
```bash
# Check token validity
make hf-status

# Regenerate token at https://huggingface.co/settings/tokens
# Request model access at model page
```

### ❌ "Access to model X is restricted"
**Solution**: Request access
1. Visit model page (e.g., `https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct`)
2. Click "Request access" 
3. Wait for approval (can take time)

### ❌ Token not working in container
**Check environment**:
```bash
# Verify token is passed to container
docker run --rm -e HF_TOKEN=$HF_TOKEN glitcher:gpu \
  python -c "import os; print('Token set:', 'HF_TOKEN' in os.environ)"
```

## Environment File (.env)

Create `.env` file in project root:
```bash
# GPU Configuration
CUDA_VISIBLE_DEVICES=0

# Directories
MODELS_PATH=./models
OUTPUTS_PATH=./outputs
DATA_PATH=./data

# Hugging Face Authentication
HF_TOKEN=hf_your_token_here
```

## Model Access Matrix

| Model | Size | Auth Required | Access Time |
|-------|------|---------------|-------------|
| **Llama 3.2 1B** | 1B | ✅ Required | ~1 hour |
| **Llama 3.2 3B** | 3B | ✅ Required | ~1 hour |
| **Llama 3.1 8B** | 8B | ✅ Required | ~1 hour |
| **Mistral 7B** | 7B | ✅ Required | ~1 hour |
| **DialoGPT** | 355M | ❌ Public | Instant |

## Advanced Usage

### Multiple Tokens
```bash
# Different tokens for different models
export HF_TOKEN_LLAMA=hf_token_for_llama
export HF_TOKEN_MISTRAL=hf_token_for_mistral

# Use specific token
HF_TOKEN=$HF_TOKEN_LLAMA make mine MODEL=meta-llama/Llama-3.2-1B-Instruct
```

### Token Security
```bash
# Store token securely (don't commit to git)
echo ".env" >> .gitignore

# Use different tokens for different environments
cp .env.example .env
# Edit .env with your tokens
```

### Batch Operations
```bash
# Mine multiple models with authentication
for model in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
  HF_TOKEN=$HF_TOKEN make mine MODEL=$model
done
```

## Support

### Get Help
- **Authentication Issues**: `make hf-status`
- **Model Access**: Visit model pages on Hugging Face
- **Technical Issues**: Check [Glitcher Documentation](README-Docker.md)

### Useful Links
- [HF Token Settings](https://huggingface.co/settings/tokens)
- [Llama 3.2 Models](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [Model Access Requests](https://huggingface.co/docs/hub/models-gated)

---

**🎉 Ready to Go!** 

Once authentication is working, you can access state-of-the-art models:
```bash
HF_TOKEN=your_token make mine MODEL=meta-llama/Llama-3.2-1B-Instruct
```
