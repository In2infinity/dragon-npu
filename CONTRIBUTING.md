# Contributing to DragonNPU

First off, thank you for considering contributing to DragonNPU! We're building the future of AI acceleration on Linux, and every contribution helps.

## 🚀 How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in [Issues](https://github.com/dragonfire/dragon-npu/issues)
- Create a detailed bug report including:
  - NPU hardware (vendor, model)
  - Linux kernel version
  - Python version
  - Steps to reproduce
  - Expected vs actual behavior

### Suggesting Enhancements
- Open an issue with the `enhancement` label
- Describe the feature and its use case
- Explain how it benefits NPU acceleration

### Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`python3 -m pytest tests/`)
5. Commit with descriptive message
6. Push to your fork
7. Open a Pull Request

## 🏗️ Development Setup

```bash
# Clone repository
git clone https://github.com/dragonfire/dragon-npu.git
cd dragon-npu

# Install in development mode
pip install -e ".[dev]"

# Run tests
python3 -m pytest tests/

# Run benchmarks
python3 dragon_npu_cli.py benchmark
```

## 📝 Code Style

- Follow PEP 8
- Use type hints where possible
- Add docstrings to all functions/classes
- Keep functions focused and small
- Comment complex NPU-specific logic

## 🧪 Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Include performance benchmarks for NPU operations

## 📚 Documentation

- Update README.md for user-facing changes
- Add docstrings for new APIs
- Include examples for new features

## 🎯 Priority Areas

We especially welcome contributions in:
- Support for new NPU vendors (Intel VPU, Qualcomm, etc.)
- Model optimization techniques
- Performance improvements
- Documentation and examples
- Cross-platform compatibility

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Celebrate the pioneering spirit of open-source NPU development

## 📬 Contact

- GitHub Issues: [Issues](https://github.com/dragonfire/dragon-npu/issues)
- Discussions: [Discussions](https://github.com/dragonfire/dragon-npu/discussions)

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- The eternal gratitude of the Linux AI community!

---

*Together, we're democratizing AI acceleration!* 🐉🔥