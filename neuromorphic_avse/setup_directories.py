import os
from pathlib import Path

def create_directory_structure():
    """Create the directory structure for the neuromorphic AVSE project."""
    
    # Define the directory structure
    directories = {
        'dataset': {
            'raw': 'Raw video and audio files',
            'processed': 'Preprocessed data',
            'features': 'Extracted features',
            'samples': 'Sample videos for testing'
        },
        'models': {
            'checkpoints': 'Model checkpoints and saved weights',
            'configs': 'Model configuration files'
        },
        'utils': {
            'audio': 'Audio processing utilities',
            'visual': 'Visual processing utilities',
            'neuromorphic': 'SNN simulation tools'
        },
        'results': {
            'enhanced_audio': 'Enhanced speech outputs',
            'visualizations': 'Lip tracking visualizations',
            'metrics': 'Performance metrics and evaluations'
        },
        'tests': {
            'unit': 'Unit tests',
            'integration': 'Integration tests'
        },
        'docs': {
            'api': 'API documentation',
            'examples': 'Usage examples'
        }
    }
    
    # Create directories and README files
    for main_dir, subdirs in directories.items():
        # Create main directory
        os.makedirs(main_dir, exist_ok=True)
        
        # Create README for main directory
        with open(os.path.join(main_dir, 'README.md'), 'w') as f:
            f.write(f"# {main_dir.capitalize()}\n\n")
            f.write(f"Directory for {main_dir} related files and subdirectories.\n\n")
            f.write("## Subdirectories\n")
            for subdir, description in subdirs.items():
                f.write(f"- `{subdir}/`: {description}\n")
        
        # Create subdirectories
        for subdir in subdirs:
            path = os.path.join(main_dir, subdir)
            os.makedirs(path, exist_ok=True)
            
            # Create README for subdirectory
            with open(os.path.join(path, 'README.md'), 'w') as f:
                f.write(f"# {subdir.capitalize()}\n\n")
                f.write(f"{subdirs[subdir]}\n")
    
    # Create additional important files
    files_to_create = {
        '.gitignore': """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
dataset/raw/*
dataset/processed/*
dataset/features/*
!dataset/raw/.gitkeep
!dataset/processed/.gitkeep
!dataset/features/.gitkeep

models/checkpoints/*
!models/checkpoints/.gitkeep

results/*
!results/.gitkeep
""",
        'setup.py': """
from setuptools import setup, find_packages

setup(
    name="neuromorphic_avse",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'librosa>=0.9.0',
        'opencv-python>=4.5.0',
        'moviepy>=1.0.3',
        'torch>=1.9.0',
        'torchaudio>=0.9.0',
        'snntorch>=0.5.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0',
        'tqdm>=4.62.0',
        'scikit-learn>=0.24.0',
        'soundfile>=0.10.0',
        'pydub>=0.25.0',
        'dlib>=19.22.0',
        'face-alignment>=1.3.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Neuromorphic Audio-Visual Speech Enhancement",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neuromorphic_avse",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
""",
        'LICENSE': """
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    }
    
    # Create additional files
    for filename, content in files_to_create.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
    
    # Create .gitkeep files in empty directories
    for dirpath in [Path('dataset/raw'), Path('dataset/processed'), 
                   Path('dataset/features'), Path('models/checkpoints'),
                   Path('results')]:
        (dirpath / '.gitkeep').touch()
    
    print("Directory structure created successfully!")
    print("\nProject structure:")
    for main_dir, subdirs in directories.items():
        print(f"\n{main_dir}/")
        for subdir in subdirs:
            print(f"  └── {subdir}/")

if __name__ == "__main__":
    create_directory_structure() 