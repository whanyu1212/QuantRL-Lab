#!/bin/bash

# Release automation script for QuantRL-Lab
# Usage: ./release.sh [patch|minor|major]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if version type is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Please specify version bump type (patch, minor, or major)${NC}"
    echo "Usage: ./release.sh [patch|minor|major]"
    echo ""
    echo "Examples:"
    echo "  ./release.sh patch   # 0.1.0 -> 0.1.1 (bug fixes)"
    echo "  ./release.sh minor   # 0.1.1 -> 0.2.0 (new features)"
    echo "  ./release.sh major   # 0.2.0 -> 1.0.0 (breaking changes)"
    exit 1
fi

VERSION_TYPE=$1

# Validate version type
if [[ ! "$VERSION_TYPE" =~ ^(patch|minor|major)$ ]]; then
    echo -e "${RED}Error: Invalid version type '$VERSION_TYPE'${NC}"
    echo "Must be one of: patch, minor, major"
    exit 1
fi

echo -e "${YELLOW}QuantRL-Lab Release Automation${NC}"
echo "=================================="
echo ""

# Check if we're on main branch
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${YELLOW}Warning: You're on branch '$CURRENT_BRANCH', not 'main'${NC}"
    read -p "Do you want to continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${RED}Error: You have uncommitted changes${NC}"
    echo "Please commit or stash your changes before releasing"
    git status -s
    exit 1
fi

# Pull latest changes
echo -e "${GREEN}Pulling latest changes...${NC}"
git pull origin $CURRENT_BRANCH

# Get current version
CURRENT_VERSION=$(poetry version -s)
echo "Current version: $CURRENT_VERSION"

# Run tests
echo ""
echo -e "${GREEN}Running tests...${NC}"
if ! poetry run pytest tests/; then
    echo -e "${RED}Tests failed! Aborting release.${NC}"
    exit 1
fi

# Run linting
echo ""
echo -e "${GREEN}Running linting...${NC}"
poetry run black --check src tests || {
    echo -e "${YELLOW}Code formatting issues found. Running black...${NC}"
    poetry run black src tests
}

poetry run isort --check-only src tests || {
    echo -e "${YELLOW}Import sorting issues found. Running isort...${NC}"
    poetry run isort src tests
}

if ! poetry run flake8 src tests; then
    echo -e "${RED}Linting failed! Please fix the issues and try again.${NC}"
    exit 1
fi

# Bump version
echo ""
echo -e "${GREEN}Bumping version ($VERSION_TYPE)...${NC}"
poetry version $VERSION_TYPE
NEW_VERSION=$(poetry version -s)
echo "New version: $NEW_VERSION"

# Commit version bump
echo ""
echo -e "${GREEN}Committing version bump...${NC}"
git add pyproject.toml
git commit -m "Bump version to $NEW_VERSION"

# Create tag
TAG_NAME="v$NEW_VERSION"
echo ""
echo -e "${GREEN}Creating tag $TAG_NAME...${NC}"
git tag -a "$TAG_NAME" -m "Release version $NEW_VERSION"

# Push changes
echo ""
echo -e "${YELLOW}Ready to push changes and tag to remote${NC}"
echo "This will:"
echo "  1. Push commit to $CURRENT_BRANCH"
echo "  2. Push tag $TAG_NAME"
echo "  3. Trigger CD pipeline to publish to PyPI"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Aborted. To undo local changes:${NC}"
    echo "  git reset --hard HEAD~1"
    echo "  git tag -d $TAG_NAME"
    exit 1
fi

echo -e "${GREEN}Pushing to remote...${NC}"
git push origin $CURRENT_BRANCH
git push origin $TAG_NAME

echo ""
echo -e "${GREEN}✓ Release process complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Go to GitHub and create a release for tag $TAG_NAME"
echo "     https://github.com/whanyu1212/QuantRL-Lab/releases/new?tag=$TAG_NAME"
echo "  2. Add release notes describing the changes"
echo "  3. Publish the release"
echo "  4. The CD pipeline will automatically publish to PyPI"
echo ""
echo "Or the CD pipeline will run automatically since the tag was pushed."
echo ""
echo "Monitor the workflow at:"
echo "  https://github.com/whanyu1212/QuantRL-Lab/actions"
