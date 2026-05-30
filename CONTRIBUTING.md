# Contributing to MedQueryAI 🏥

Thank you for your interest in contributing to MedQueryAI! This guide will help you get started.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/MediaQueryAI.git`
3. **Install** dependencies: `pip install -r requirements.txt`
4. **Create** a branch: `git checkout -b feature/your-feature`

## Development Setup

```bash
# Copy environment template
cp .env.example .env

# Add your API key (Gemini is free!)
# Edit .env with your key

# Run the app
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

## Code Style

- Use clear, descriptive variable names
- Add docstrings to all functions
- Keep functions focused and small
- Follow PEP 8 guidelines

## Pull Request Process

1. Update documentation if you change any functionality
2. Add tests for new features
3. Ensure all tests pass before submitting
4. Write a clear PR description explaining your changes

## Areas We Need Help

- 📄 DICOM / HL7 FHIR document support
- 🔀 Query routing by medical category
- 🌐 REST API endpoint
- 🧠 Conversation memory with follow-up questions
- 📊 More embedding model options
- 🧪 Additional test coverage

## Questions?

Open an issue or start a discussion — we're happy to help!
