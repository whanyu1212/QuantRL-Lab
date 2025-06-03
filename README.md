# QuantRL-Lab
A Python testbed for Reinforcement Learning in finance

### Setup Guide

1. Clone the Repository
```bash
git clone https://github.com/whanyu1212/QuantRL-Lab.git
```

2. Install Poetry for dependency management
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Sync dependencies (It also installs the current project in dev mode)
```bash
poetry install
```

4. Activate virtual environment (Note that the `shell` command is deprecated in the latest poetry version)
```bash
poetry env activate
# a venv path will be printed in the terminal, just copy and run it
# e.g.,
source /home/codespace/.cache/pypoetry/virtualenvs/multi-agent-quant-cj6_z41n-py3.12/bin/activate
```

5. Install jupyter kernel
```bash
# You can change the name and display name according to your preference
python -m ipykernel install --user --name multi-agent-quant --display-name "Multi Agent Quant"
```

6. Set up environment variables
```bash
# Copy the example environment file
cp .env.example .env

# Open .env file and replace the placeholder values with your actual credentials
# You can use any text editor, here using VS Code
code .env
```

Make sure to replace all placeholder values in the `.env` file with your actual API keys and credentials. Never commit the `.env` file to version control.

<br>

7. Set up pre-commit hooks
```bash
# Install pre-commit
poetry add pre-commit

# Install the git hooks
pre-commit install

# Optional: run pre-commit on all files
pre-commit run --all-files
```

The pre-commit hooks will check for:
- Code formatting (black)
- Import sorting (isort)
- Code linting (flake8)
- Docstring formatting (docformatter)
- Basic file checks (trailing whitespace, YAML validation, etc.)

To skip pre-commit hooks temporarily:
```bash
git commit -m "your message" --no-verify
```

For more details, please refer to `.pre-commit-config.yaml` file.
