"""Validation script to verify project setup"""

import sys
from pathlib import Path


def check_directory_structure():
    """Check that all required directories exist"""
    required_dirs = [
        'src',
        'tests',
        'tests/unit',
        'configs',
        'data',
        'output',
    ]
    
    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_required_files():
    """Check that all required files exist"""
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/logging_config.py',
        'tests/__init__.py',
        'tests/unit/__init__.py',
        'tests/unit/test_config.py',
        'tests/unit/test_logging.py',
        'configs/default_config.yaml',
        'requirements.txt',
        'pytest.ini',
        '.gitignore',
    ]
    
    print("\nChecking required files...")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist


def check_config_file():
    """Check that config file is valid YAML"""
    print("\nChecking configuration file...")
    try:
        import yaml
        config_path = Path('configs/default_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = [
            'ed_parameters',
            'qvae_architecture',
            'training',
            'analysis',
            'paths',
            'logging'
        ]
        
        all_sections = True
        for section in required_sections:
            if section in config:
                print(f"  ✓ {section}")
            else:
                print(f"  ✗ {section} - MISSING")
                all_sections = False
        
        return all_sections
    except ImportError:
        print("  ⚠ PyYAML not installed - skipping config validation")
        return True
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        return False


def check_imports():
    """Check if core modules can be imported"""
    print("\nChecking module imports...")
    
    # Add src to path
    sys.path.insert(0, str(Path.cwd()))
    
    modules_to_check = [
        ('src.config', 'Config'),
        ('src.logging_config', 'setup_logging'),
    ]
    
    all_imported = True
    for module_name, class_name in modules_to_check:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✓ {module_name}.{class_name}")
        except ImportError as e:
            print(f"  ⚠ {module_name}.{class_name} - Import error (may need dependencies): {e}")
            # Don't fail on import errors - dependencies may not be installed yet
        except Exception as e:
            print(f"  ✗ {module_name}.{class_name} - Error: {e}")
            all_imported = False
    
    return all_imported


def main():
    """Run all validation checks"""
    print("=" * 60)
    print("J1-J2 Heisenberg Prometheus - Setup Validation")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Required Files", check_required_files),
        ("Configuration File", check_config_file),
        ("Module Imports", check_imports),
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"\n✗ {check_name} failed with error: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: pytest tests/unit/")
        print("3. See SETUP.md for more information")
        return 0
    else:
        print("\n✗ Some checks failed. Please review the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
