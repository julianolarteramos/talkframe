#!/bin/bash
# Copy this case's fixtures into the empty run workspace (cwd). $0 is this script, inside the case dir.
set -e
mkdir -p fixtures
cp "$(dirname "$0")"/fixtures/* fixtures/
