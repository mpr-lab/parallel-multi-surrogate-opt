## Architecture Diagram

<img width="1241" height="8191" alt="Parallel multi-surrogate BO architecture" src="https://github.com/user-attachments/assets/7676c0cf-9a1a-466a-80ac-293d0fdedad1" />

## Setting Up the Development Environment

Clone the repository:

```bash
git clone https://github.com/mpr-lab/parallel-multi-surrogate-opt.git
cd parallel-multi-surrogate-opt

Create and actvate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

Install the project dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

Run the full test suite:
```bash
python -m pytest -q

Reactivate the environment via:
source .venv/bin/activate

