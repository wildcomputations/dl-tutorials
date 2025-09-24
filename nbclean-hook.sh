#!/bin/bash
set -e

echo "Running nbclean on staged notebooks..."

for nb in $(git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$' || true); do
  nbclean clean "$nb" --output "$nb" --preserve-output-tag keep_output
  git add "$nb"
done
