#!/bin/bash
#
# Overleaf Paper Synchronization Script
#
# This script synchronizes the Overleaf paper content with the local repository
# using git subtree. It allows bidirectional sync between the codebase and paper.
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the correct directory
if [ ! -f "README.md" ] || [ ! -d "project" ]; then
    print_error "Please run this script from the infant-fmri repository root directory"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  pull     - Pull latest changes from Overleaf to local paper/ directory"
    echo "  push     - Push local paper changes to Overleaf"
    echo "  status   - Show synchronization status"
    echo "  setup    - Initial setup (already done)"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 pull     # Sync latest paper changes from Overleaf"
    echo "  $0 push     # Push local paper edits to Overleaf"
    echo "  $0 status   # Check sync status"
}

# Function to pull from Overleaf
pull_from_overleaf() {
    print_status "Fetching latest changes from Overleaf..."

    # Fetch latest changes
    git fetch overleaf master

    # Check if there are new changes
    local_commit=$(git subtree split --prefix=paper)
    remote_commit=$(git rev-parse overleaf/master)

    if [ "$local_commit" = "$remote_commit" ]; then
        print_status "Paper is already up to date"
        return 0
    fi

    print_status "Pulling latest paper changes from Overleaf..."
    git subtree pull --prefix=paper overleaf master --squash -m "Sync latest paper changes from Overleaf"

    print_status "Successfully synchronized paper from Overleaf"
    print_warning "Please review the changes before committing to the main repository"
}

# Function to push to Overleaf
push_to_overleaf() {
    print_status "Pushing local paper changes to Overleaf..."

    # Check if there are uncommitted changes in paper/
    if [ -n "$(git status --porcelain paper/)" ]; then
        print_error "You have uncommitted changes in paper/. Please commit them first:"
        git status --porcelain paper/
        exit 1
    fi

    # Push changes to Overleaf
    git subtree push --prefix=paper overleaf master

    print_status "Successfully pushed paper changes to Overleaf"
    print_status "Changes are now available in your Overleaf project"
}

# Function to show status
show_status() {
    print_status "Paper Synchronization Status"
    echo ""

    # Show git remotes
    echo -e "${BLUE}Git Remotes:${NC}"
    git remote -v | grep -E "(origin|overleaf)"
    echo ""

    # Show paper directory status
    echo -e "${BLUE}Paper Directory Status:${NC}"
    if [ -d "paper" ]; then
        echo "✓ Paper directory exists"
        echo "✓ Paper files: $(ls paper/ | wc -l) items"
        if [ -d "paper/img" ]; then
            echo "✓ Images directory: $(ls paper/img/ | wc -l) images"
        fi
    else
        echo "✗ Paper directory missing"
    fi
    echo ""

    # Show git status for paper files
    echo -e "${BLUE}Paper Changes:${NC}"
    git status --porcelain paper/ || echo "No changes in paper directory"
    echo ""

    # Show last sync info
    echo -e "${BLUE}Last Sync:${NC}"
    git log --oneline --grep="Overleaf\|paper" -n 3 || echo "No sync history found"
}

# Function for initial setup (already done, but documented)
setup_sync() {
    print_status "Overleaf sync is already set up!"
    echo ""
    echo "Current configuration:"
    echo "  - Overleaf remote: $(git remote get-url overleaf 2>/dev/null || echo 'Not configured')"
    echo "  - Paper directory: paper/"
    echo "  - Sync method: Git subtree"
    echo ""
    echo "To reconfigure, you can:"
    echo "  1. Remove remote: git remote remove overleaf"
    echo "  2. Remove paper dir: rm -rf paper/"
    echo "  3. Re-add remote: git remote add overleaf https://git.overleaf.com/YOUR_PROJECT_ID"
    echo "  4. Re-sync: git subtree add --prefix=paper overleaf master --squash"
}

# Main command handling
case "${1:-help}" in
    "pull")
        pull_from_overleaf
        ;;
    "push")
        push_to_overleaf
        ;;
    "status")
        show_status
        ;;
    "setup")
        setup_sync
        ;;
    "help"|*)
        show_usage
        ;;
esac