# Vast.ai Local Model Upload Runbook

This runbook assumes the server has no reliable outbound access to Hugging Face.
Download the model on the local PC, upload it to Vast, and point `SNU_MODEL` at the uploaded folder.

## 1. Local: download the Hugging Face model

Run in local PowerShell from this repo folder:

```powershell
cd "C:\Users\82103\OneDrive\바탕 화면\SNUAI_챌린지\snu-ai-challenge-donothangsha\dgddgd314"
.\venv\Scripts\activate
pip install -U huggingface_hub
```

Set your Hugging Face token for this PowerShell session:

```powershell
$env:HF_TOKEN="hf_your_token_here"
$env:HF_HUB_DISABLE_XET="1"
$env:HF_HUB_ETAG_TIMEOUT="60"
$env:HF_HUB_DOWNLOAD_TIMEOUT="600"
```

Download and archive PaliGemma:

```powershell
python scripts\download_hf_model.py --repo-id google/paligemma-3b-pt-448
```

Expected local outputs:

```text
local_models\paligemma-3b-pt-448\
local_models\paligemma-3b-pt-448.tar
```

## 2. Local: prepare the dataset archive

If not already created:

```powershell
cd "C:\Users\82103\OneDrive\바탕 화면\SNUAI_챌린지\snu-ai-challenge-donothangsha\dgddgd314"
tar -cf snu_data.tar data\snuaichallenge_data
```

## 3. Vast: create and test the instance

Use a PyTorch/CUDA template and a GPU with at least 24 GB VRAM.

After connecting with SSH:

```bash
nvidia-smi
cd /workspace
apt-get update
apt-get install -y git
git clone -b dgddgd314/gemma <your-github-repo-url> project
cd /workspace/project/dgddgd314
```

## 4. Local: upload dataset and model archive

Use the latest Vast SSH command values.

If Vast shows:

```text
ssh -p 11397 root@198.2.214.6 -L 8080:localhost:8080
```

upload with:

```powershell
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P 11397 .\snu_data.tar root@198.2.214.6:/workspace/snu_data.tar
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P 11397 .\local_models\paligemma-3b-pt-448.tar root@198.2.214.6:/workspace/paligemma-3b-pt-448.tar
```

If Vast only works through proxy and shows:

```text
ssh -p 17489 root@ssh3.vast.ai -L 8080:localhost:8080
```

upload with:

```powershell
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P 17489 .\snu_data.tar root@ssh3.vast.ai:/workspace/snu_data.tar
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P 17489 .\local_models\paligemma-3b-pt-448.tar root@ssh3.vast.ai:/workspace/paligemma-3b-pt-448.tar
```

## 5. Vast: unpack dataset and model

```bash
cd /workspace/project/dgddgd314
mkdir -p data /workspace/models
tar -xf /workspace/snu_data.tar
tar -xf /workspace/paligemma-3b-pt-448.tar -C /workspace/models
ls data/snuaichallenge_data
ls /workspace/models/paligemma-3b-pt-448
```

## 6. Vast: install Python dependencies

```bash
cd /workspace/project/dgddgd314
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pandas pillow tqdm transformers peft accelerate bitsandbytes
```

Check CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 7. Vast: run smoke test and training

Use the local uploaded model path:

```bash
export SNU_DATA_DIR=/workspace/project/dgddgd314/data/snuaichallenge_data
export SNU_MODEL=/workspace/models/paligemma-3b-pt-448
export SNU_ADAPTER_DIR=/workspace/project/dgddgd314/outputs/paligemma_lora
export SNU_CONTACT_SHEET_SIZE=448
```

Run:

```bash
python scripts/paligemma_smoke.py
python scripts/paligemma_train_skeleton.py --max-steps 10
python scripts/paligemma_train_skeleton.py --max-steps 100
python scripts/paligemma_infer.py --limit 20 --out outputs/paligemma_probe.csv
python scripts/validate_submission.py --data-dir "$SNU_DATA_DIR" --submission outputs/paligemma_probe.csv
```

After confirming the archives were unpacked:

```bash
rm /workspace/snu_data.tar
rm /workspace/paligemma-3b-pt-448.tar
```

## Error Cases

If `scp` fails with SFTP/subsystem errors, retry with `-O`:

```powershell
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P <port> <local-file> root@<host>:/workspace/
```

If direct SSH fails, use the proxy command shown by Vast:

```powershell
scp -O -i $env:USERPROFILE\.ssh\id_ed25519 -P <proxy-port> <local-file> root@ssh3.vast.ai:/workspace/
```

If `SNU_MODEL` still tries to contact Hugging Face, verify it is a local path:

```bash
echo "$SNU_MODEL"
ls "$SNU_MODEL"
```

If CUDA is not visible:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If disk is tight:

```bash
df -h /workspace
du -sh /workspace/*
```
