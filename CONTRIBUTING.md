# Contributing to QuantRL-Lab

Thank you for your interest in contributing to QuantRL-Lab! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)
- [Getting Help](#getting-help)

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone. Please:

- Be respectful and constructive in discussions
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Poetry for dependency management
- Git for version control
- Familiarity with reinforcement learning concepts (helpful but not required)

### Areas Where You Can Contribute

We welcome contributions in several areas:

- 🐛 **Bug fixes**: Fix issues reported in the issue tracker
- ✨ **New features**: Implement new trading environments, strategies, or indicators
- 📚 **Documentation**: Improve or add documentation, examples, tutorials
- 🧪 **Tests**: Add test coverage or improve existing tests
- 🎨 **Code quality**: Refactoring, performance improvements
- 💡 **Ideas**: Propose new features or improvements in discussions

## Development Setup

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/QuantRL-Lab.git
   cd QuantRL-Lab
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/whanyu1212/QuantRL-Lab.git
   ```

3. **Install Poetry**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4. **Install dependencies**
   ```bash
   poetry install
   ```

5. **Activate virtual environment**
   ```bash
   poetry shell
   ```

6. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

7. **Create a branch for your work**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check the [issue tracker](https://github.com/whanyu1212/QuantRL-Lab/issues) for existing reports
2. Try to reproduce the bug with the latest version
3. Gather relevant information (Python version, OS, error messages, etc.)

Create a bug report including:
- Clear, descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Code samples or error messages
- Environment details (Python version, OS, package versions)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. Include:
- Clear, descriptive title
- Detailed description of the proposed feature
- Use cases and motivation
- Examples of how the feature would work
- Any potential drawbacks or alternatives considered

### Pull Requests

1. **Discuss first** for large changes
   - Open an issue to discuss your idea before writing code
   - This prevents duplicate work and ensures alignment

2. **Keep changes focused**
   - One feature/fix per pull request
   - Avoid mixing refactoring with new features

3. **Write good commit messages**
   ```
   Short (50 chars or less) summary

   More detailed explanatory text, if necessary. Wrap it to
   about 72 characters. The blank line separating the summary
   from the body is critical.

   Explain the problem that this commit is solving. Focus on
   why you are making this change as opposed to how.

   - Bullet points are okay
   - Use imperative mood ("Add feature" not "Added feature")
   ```

## Coding Standards

### Code Style

We follow PEP 8 with some modifications:
- Line length: 120 characters (configured in black)
- Use type hints where appropriate
- Write docstrings for all public functions, classes, and modules

### Code Formatting

We use automated tools to maintain code quality:

```bash
# Format code with black
poetry run black src tests

# Sort imports with isort
poetry run isort src tests

# Lint with flake8
poetry run flake8 src tests

# Type checking (if applicable)
poetry run mypy src
```

**Pre-commit hooks will automatically run these checks** before each commit.

### Docstring Format

Use Google-style docstrings:

```python
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe ratio for a series of returns.

    Args:
        returns: Array of portfolio returns
        risk_free_rate: Risk-free rate for Sharpe calculation. Defaults to 0.0.

    Returns:
        The calculated Sharpe ratio.

    Raises:
        ValueError: If returns array is empty or contains invalid values.

    Example:
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03])
        >>> sharpe = calculate_sharpe_ratio(returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if len(returns) == 0:
        raise ValueError("Returns array cannot be empty")

    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)
```

### Project Structure

When adding new features, follow the existing structure:

```
src/quantrl_lab/
├── custom_envs/          # Trading environments
│   ├── stock/           # Stock trading specific
│   ├── crypto/          # Crypto trading specific
│   └── fx/              # Forex trading specific
├── backtesting/         # Backtesting framework
├── data/                # Data loading and processing
├── feature_selection/   # Feature engineering
├── screener/            # Stock screening
├── trading/             # Live trading utilities
└── utils/               # Shared utilities

tests/                   # Mirror src structure
```

## Testing

### Running Tests

```bash
# Run all tests
poetry run pytest tests/

# Run with coverage
poetry run pytest tests/ --cov=src/quantrl_lab --cov-report=html

# Run specific test file
poetry run pytest tests/test_env.py

# Run specific test
poetry run pytest tests/test_env.py::test_function_name
```

### Writing Tests

- Write tests for all new features and bug fixes
- Aim for high test coverage (>80%)
- Use descriptive test names: `test_portfolio_rebalancing_with_transaction_costs`
- Use fixtures for common setup
- Test edge cases and error conditions

Example test structure:

```python
import pytest
import numpy as np
from quantrl_lab.custom_envs.stock import SingleStockTradingEnv


class TestSingleStockTradingEnv:
    """Test suite for SingleStockTradingEnv."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for testing."""
        return np.random.rand(100, 5).astype(np.float32)

    def test_environment_initialization(self, sample_data):
        """Test that environment initializes correctly."""
        env = SingleStockTradingEnv(data=sample_data)
        assert env is not None
        assert env.data.shape == sample_data.shape

    def test_reset_returns_valid_observation(self, sample_data):
        """Test that reset returns observation of correct shape."""
        env = SingleStockTradingEnv(data=sample_data)
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
```

### Test Coverage

- All new code should have tests
- Aim for at least 80% code coverage
- Critical paths should have 100% coverage
- Check coverage report: `open htmlcov/index.html`

## Submitting Changes

### Before Submitting

Checklist before submitting a pull request:

- [ ] Code follows the project's style guidelines
- [ ] All tests pass locally
- [ ] New tests added for new features
- [ ] Documentation updated (if applicable)
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Pre-commit hooks pass
- [ ] Branch is up to date with main

### Pull Request Process

1. **Update your branch**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create pull request on GitHub**
   - Use a clear, descriptive title
   - Reference related issues (e.g., "Fixes #123")
   - Describe what changed and why
   - Include screenshots for UI changes
   - List any breaking changes

4. **Pull request template**
   ```markdown
   ## Description
   Brief description of the changes

   ## Motivation and Context
   Why is this change needed? What problem does it solve?

   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that breaks existing functionality)
   - [ ] Documentation update

   ## How Has This Been Tested?
   Describe the tests you ran

   ## Checklist
   - [ ] My code follows the project's style guidelines
   - [ ] I have performed a self-review of my code
   - [ ] I have commented my code, particularly in hard-to-understand areas
   - [ ] I have made corresponding changes to the documentation
   - [ ] My changes generate no new warnings
   - [ ] I have added tests that prove my fix is effective or that my feature works
   - [ ] New and existing unit tests pass locally with my changes
   - [ ] Any dependent changes have been merged and published
   ```

5. **Respond to review feedback**
   - Be open to feedback and suggestions
   - Make requested changes promptly
   - Ask for clarification if needed
   - Update your branch as needed

6. **After approval**
   - Maintainer will merge your PR
   - Your contribution will be in the next release!

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run `./release.sh patch|minor|major`
4. Create GitHub Release
5. CD workflow automatically publishes to PyPI

See [docs/PUBLISHING_CHECKLIST.md](docs/PUBLISHING_CHECKLIST.md) for details.

## Development Workflow

### Typical workflow for contributors:

```bash
# 1. Sync with upstream
git checkout main
git pull upstream main

# 2. Create feature branch
git checkout -b feature/amazing-new-feature

# 3. Make changes and test
# ... code code code ...
poetry run pytest tests/

# 4. Commit with pre-commit hooks
git add .
git commit -m "Add amazing new feature"

# 5. Push to your fork
git push origin feature/amazing-new-feature

# 6. Create pull request on GitHub
```

### Keeping your fork updated

```bash
# Fetch upstream changes
git fetch upstream

# Update main branch
git checkout main
git merge upstream/main
git push origin main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main
```

## Getting Help

### Resources

- 📖 [Documentation](https://github.com/whanyu1212/QuantRL-Lab/blob/main/README.md)
- 🐛 [Issue Tracker](https://github.com/whanyu1212/QuantRL-Lab/issues)
- 💬 [Discussions](https://github.com/whanyu1212/QuantRL-Lab/discussions)

### Questions?

- Check existing [issues](https://github.com/whanyu1212/QuantRL-Lab/issues) and [discussions](https://github.com/whanyu1212/QuantRL-Lab/discussions)
- Open a new [discussion](https://github.com/whanyu1212/QuantRL-Lab/discussions) for general questions
- Open an [issue](https://github.com/whanyu1212/QuantRL-Lab/issues) for bugs or feature requests

## Recognition

Contributors will be:
- Listed in the project's contributors page
- Mentioned in release notes for significant contributions
- Appreciated and thanked! 🙏

## License

By contributing to QuantRL-Lab, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to QuantRL-Lab! Your efforts help make this project better for everyone. 🚀
