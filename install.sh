#!/bin/bash
# DragonNPU Installation Script
# Complete setup for DragonNPU on Linux systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ASCII Art Banner
show_banner() {
    echo -e "${CYAN}"
    cat << "EOF"
    ____                              _   _ ____  _   _ 
   |  _ \ _ __ __ _  __ _  ___  _ __ | \ | |  _ \| | | |
   | | | | '__/ _` |/ _` |/ _ \| '_ \|  \| | |_) | | | |
   | |_| | | | (_| | (_| | (_) | | | | |\  |  __/| |_| |
   |____/|_|  \__,_|\__, |\___/|_| |_|_| \_|_|    \___/ 
                    |___/                                
   🐉 Bringing AI Acceleration to Linux
EOF
    echo -e "${NC}"
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        log_error "DragonNPU requires Linux. Detected: $OSTYPE"
        exit 1
    fi
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    REQUIRED_VERSION="3.8"
    
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ required. Found: Python $PYTHON_VERSION"
        exit 1
    fi
    
    log_success "Python $PYTHON_VERSION detected"
    
    # Check for NPU hardware
    check_npu_hardware
}

# Check NPU hardware
check_npu_hardware() {
    log_info "Detecting NPU hardware..."
    
    NPU_DETECTED=false
    NPU_VENDOR="unknown"
    
    # Check for AMD XDNA
    if lsmod | grep -q "amdxdna" || [ -e "/dev/accel/accel0" ]; then
        NPU_DETECTED=true
        NPU_VENDOR="AMD XDNA"
        log_success "AMD XDNA NPU detected"
    # Check for Intel VPU
    elif lspci | grep -qi "vpu\|neural"; then
        NPU_DETECTED=true
        NPU_VENDOR="Intel VPU"
        log_success "Intel VPU detected"
    # Check for Qualcomm
    elif [ -e "/dev/ion" ] && [ -e "/dev/adsprpc-smd" ]; then
        NPU_DETECTED=true
        NPU_VENDOR="Qualcomm Hexagon"
        log_success "Qualcomm Hexagon NPU detected"
    # Check for Rockchip
    elif [ -e "/dev/rknpu" ]; then
        NPU_DETECTED=true
        NPU_VENDOR="Rockchip NPU"
        log_success "Rockchip NPU detected"
    else
        log_warning "No NPU hardware detected. DragonNPU will run in simulation mode."
    fi
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies..."
    
    # Detect package manager
    if command -v apt &> /dev/null; then
        PKG_MANAGER="apt"
        INSTALL_CMD="sudo apt install -y"
    elif command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        INSTALL_CMD="sudo dnf install -y"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
        INSTALL_CMD="sudo pacman -S --noconfirm"
    else
        log_warning "Unknown package manager. Please install dependencies manually."
        return
    fi
    
    log_info "Using package manager: $PKG_MANAGER"
    
    # Common dependencies
    DEPS="python3-pip python3-venv python3-dev build-essential"
    
    if [ "$PKG_MANAGER" = "apt" ]; then
        sudo apt update
        $INSTALL_CMD $DEPS python3-numpy
    elif [ "$PKG_MANAGER" = "dnf" ]; then
        $INSTALL_CMD $DEPS python3-numpy
    elif [ "$PKG_MANAGER" = "pacman" ]; then
        $INSTALL_CMD base-devel python python-pip python-numpy
    fi
    
    log_success "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Create virtual environment
    VENV_DIR="$HOME/.dragon-npu-env"
    
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        log_success "Virtual environment created at $VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install DragonNPU
    pip install -e . || pip install .
    
    # Install optional dependencies based on NPU vendor
    if [ "$NPU_VENDOR" = "AMD XDNA" ]; then
        log_info "Installing AMD XDNA specific dependencies..."
        # Install IRON API if available
    elif [ "$NPU_VENDOR" = "Intel VPU" ]; then
        log_info "Installing Intel VPU specific dependencies..."
        pip install openvino 2>/dev/null || true
    fi
    
    log_success "Python dependencies installed"
}

# Setup NPU drivers
setup_npu_drivers() {
    log_info "Setting up NPU drivers..."
    
    if [ "$NPU_VENDOR" = "AMD XDNA" ]; then
        setup_amd_xdna
    elif [ "$NPU_VENDOR" = "Intel VPU" ]; then
        setup_intel_vpu
    elif [ "$NPU_VENDOR" = "Qualcomm Hexagon" ]; then
        setup_qualcomm
    elif [ "$NPU_VENDOR" = "Rockchip NPU" ]; then
        setup_rockchip
    else
        log_warning "No specific NPU driver setup needed"
    fi
}

# Setup AMD XDNA
setup_amd_xdna() {
    log_info "Setting up AMD XDNA drivers..."
    
    # Check if XRT is installed
    if [ -d "/opt/xilinx/xrt" ]; then
        log_success "XRT detected at /opt/xilinx/xrt"
    else
        log_warning "XRT not found. Please install from AMD/Xilinx website"
    fi
    
    # Load kernel module
    if ! lsmod | grep -q "amdxdna"; then
        log_info "Loading amdxdna kernel module..."
        sudo modprobe amdxdna 2>/dev/null || log_warning "Failed to load amdxdna module"
    fi
    
    # Set permissions
    if [ -e "/dev/accel/accel0" ]; then
        sudo chmod 666 /dev/accel/accel0 2>/dev/null || true
    fi
}

# Setup Intel VPU
setup_intel_vpu() {
    log_info "Setting up Intel VPU drivers..."
    # Intel VPU setup commands
}

# Setup Qualcomm
setup_qualcomm() {
    log_info "Setting up Qualcomm Hexagon drivers..."
    # Qualcomm setup commands
}

# Setup Rockchip
setup_rockchip() {
    log_info "Setting up Rockchip NPU drivers..."
    # Rockchip setup commands
}

# Create desktop entry
create_desktop_entry() {
    log_info "Creating desktop entry..."
    
    DESKTOP_FILE="$HOME/.local/share/applications/dragon-npu.desktop"
    mkdir -p "$(dirname "$DESKTOP_FILE")"
    
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=DragonNPU
Comment=AI Acceleration for Linux
Exec=$HOME/.dragon-npu-env/bin/dragon-npu
Icon=applications-science
Terminal=true
Categories=Development;Science;
EOF
    
    chmod +x "$DESKTOP_FILE"
    log_success "Desktop entry created"
}

# Setup shell integration
setup_shell_integration() {
    log_info "Setting up shell integration..."
    
    # Detect shell
    SHELL_NAME=$(basename "$SHELL")
    
    if [ "$SHELL_NAME" = "bash" ]; then
        RC_FILE="$HOME/.bashrc"
    elif [ "$SHELL_NAME" = "zsh" ]; then
        RC_FILE="$HOME/.zshrc"
    else
        RC_FILE="$HOME/.profile"
    fi
    
    # Add DragonNPU to PATH
    DRAGON_NPU_INIT="
# DragonNPU
export DRAGON_NPU_HOME=\"$HOME/.dragon-npu-env\"
alias dnpu=\"\$DRAGON_NPU_HOME/bin/dragon-npu\"
alias dragon-npu=\"\$DRAGON_NPU_HOME/bin/dragon-npu\"
"
    
    if ! grep -q "DragonNPU" "$RC_FILE" 2>/dev/null; then
        echo "$DRAGON_NPU_INIT" >> "$RC_FILE"
        log_success "Shell integration added to $RC_FILE"
    else
        log_info "Shell integration already exists"
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    source "$HOME/.dragon-npu-env/bin/activate"
    
    # Test import
    if python3 -c "import dragon_npu_core" 2>/dev/null; then
        log_success "DragonNPU core imported successfully"
    else
        log_warning "Failed to import DragonNPU core"
    fi
    
    # Test CLI
    if dragon-npu --help > /dev/null 2>&1; then
        log_success "DragonNPU CLI working"
    else
        log_warning "DragonNPU CLI not working"
    fi
    
    # Test NPU functionality
    dragon-npu status || true
}

# Main installation flow
main() {
    show_banner
    
    log_info "Starting DragonNPU installation..."
    echo ""
    
    # Check requirements
    check_requirements
    echo ""
    
    # Install dependencies
    install_system_deps
    echo ""
    
    # Install Python packages
    install_python_deps
    echo ""
    
    # Setup NPU drivers
    setup_npu_drivers
    echo ""
    
    # Create desktop entry
    create_desktop_entry
    
    # Setup shell integration
    setup_shell_integration
    echo ""
    
    # Run tests
    run_tests
    echo ""
    
    # Success message
    echo -e "${GREEN}"
    echo "=========================================="
    echo "   🎉 DragonNPU Installation Complete!    "
    echo "=========================================="
    echo -e "${NC}"
    
    echo "NPU Status:"
    echo "  Vendor: $NPU_VENDOR"
    echo "  Hardware: $([ "$NPU_DETECTED" = true ] && echo "✅ Detected" || echo "❌ Not detected")"
    echo ""
    echo "Quick Start:"
    echo "  1. Restart your shell or run: source $RC_FILE"
    echo "  2. Check NPU status: dragon-npu status"
    echo "  3. Run examples: dragon-npu test"
    echo "  4. Compile a model: dragon-npu compile model.onnx"
    echo ""
    echo "Documentation: https://github.com/dragonfire/dragon-npu"
    echo ""
    echo "🐉 Happy NPU coding!"
}

# Run main installation
main "$@"