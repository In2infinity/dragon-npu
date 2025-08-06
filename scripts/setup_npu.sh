#!/bin/bash

#============================================================================
# DragonNPU Setup Script
# Easy NPU driver installation and setup
#============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/setup_npu.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_header() {
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}              DragonNPU Setup - Dragonfire Technologies${NC}"
    echo -e "${CYAN}${BOLD}════════════════════════════════════════════════════════════════════${NC}"
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

show_usage() {
    echo "DragonNPU Setup Tool"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --setup           Run complete NPU setup"
    echo "  --status          Show current NPU status"
    echo "  --test            Test NPU functionality"
    echo "  --install-deps    Install system dependencies only"
    echo "  --install-driver  Install NPU driver only"
    echo "  --no-deps         Skip dependency installation during setup"
    echo "  --verbose         Enable verbose output"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --setup         # Complete setup"
    echo "  $0 --status        # Check current status"
    echo "  $0 --test          # Test NPU"
    echo ""
}

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    print_error "Python setup script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check Python requirements
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        return 1
    fi
    
    # Check if numpy is available
    if ! python3 -c "import numpy" &> /dev/null; then
        print_warning "NumPy not found. Installing..."
        pip3 install numpy
    fi
    
    return 0
}

# Main script logic
main() {
    print_header
    
    # Parse arguments
    case "$1" in
        --help|-h)
            show_usage
            exit 0
            ;;
        --setup)
            print_status "Starting complete DragonNPU setup..."
            check_python
            python3 "$PYTHON_SCRIPT" --setup "${@:2}"
            ;;
        --status)
            print_status "Checking DragonNPU status..."
            check_python
            python3 "$PYTHON_SCRIPT" --status "${@:2}"
            ;;
        --test)
            print_status "Testing NPU functionality..."
            check_python
            python3 "$PYTHON_SCRIPT" --test "${@:2}"
            ;;
        --install-deps)
            print_status "Installing system dependencies..."
            check_python
            python3 "$PYTHON_SCRIPT" --install-deps "${@:2}"
            ;;
        --install-driver)
            print_status "Installing NPU driver..."
            check_python
            python3 "$PYTHON_SCRIPT" --install-driver "${@:2}"
            ;;
        --no-deps)
            print_status "Running setup without dependencies..."
            check_python
            python3 "$PYTHON_SCRIPT" --setup --no-deps "${@:2}"
            ;;
        --verbose|-v)
            print_status "Running with verbose output..."
            check_python
            python3 "$PYTHON_SCRIPT" --status --verbose "${@:2}"
            ;;
        "")
            # Default: show status
            print_status "Showing current status..."
            check_python
            python3 "$PYTHON_SCRIPT" --status
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root (it will ask for sudo when needed)"
   exit 1
fi

# Run main function
main "$@"