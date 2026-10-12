## Architecture Diagram
<img width="1241" height="8191" alt="image" src="https://github.com/user-attachments/assets/7676c0cf-9a1a-466a-80ac-293d0fdedad1" />

## Setting Up Development Environment
git clone https://github.com/mpr-lab/parallel-multi-surrogate-opt.git
cd parallel-multi-surrogate-opt

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

python -m pytest -q
